def run_nomads_scheduled():
    """
    Schedules NOMADSDownloader to run at 6:00 and 12:00 São Paulo time daily, updating in-memory cache.
    """
    import pytz
    import schedule
    import threading
    from datetime import datetime, timedelta

    def job():
        logger.info("[Scheduler] Running NOMADSDownloader job for scheduled update...")
        downloader = NOMADSDownloader()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(downloader.download_forecast_range(days_ahead=5))
        downloader.cleanup_old_files(days_to_keep=30)
        logger.info("[Scheduler] NOMADSDownloader job completed.")

    def get_seconds_until(hour, minute):
        tz = pytz.timezone('America/Sao_Paulo')
        now = datetime.now(tz)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def schedule_job_at(hour, minute):
        def run_job():
            job()
            # Schedule next run
            threading.Timer(get_seconds_until(hour, minute), run_job).start()
        # Initial scheduling
        threading.Timer(get_seconds_until(hour, minute), run_job).start()

    # Schedule jobs for 6:00 and 12:00 São Paulo time
    schedule_job_at(6, 0)
    schedule_job_at(12, 0)
    logger.info("[Scheduler] NOMADSDownloader scheduled for 6:00 and 12:00 America/Sao_Paulo time.")

    # Keep main thread alive
    while True:
        time.sleep(3600)
# scheduler/nomads_daily_update.py
"""
Script automatizado para baixar dados GFS do NOMADS diariamente
Executar via cron ou scheduler do sistema
"""

import asyncio
import aiohttp
from aiohttp import ClientTimeout
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import hashlib
import schedule
import time
import threading
from typing import Optional, Dict, List
import warnings
import os
import holidays
warnings.filterwarnings('ignore')

# Garante que o diretório de logs existe
os.makedirs('NOMADS/dados/logs', exist_ok=True)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler('NOMADS/dados/logs/nomads_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NOMADSDownloader:
    def cleanup_old_forecasts(self):
        """Remove todos os arquivos forecast_*.csv exceto o mais recente na pasta de cache."""
        forecast_files = list(self.cache_dir.glob("forecast_*.csv"))
        if len(forecast_files) <= 1:
            return
        latest_file = max(forecast_files, key=lambda f: f.stat().st_mtime)
        for f in forecast_files:
            if f != latest_file:
                try:
                    f.unlink()
                except Exception as e:
                    logger.warning(f"Não foi possível remover {f}: {e}")
    """
    Classe para gerenciar downloads automatizados do NOMADS GFS
    """
    
    # In-memory cache for latest consolidated forecast
    _latest_forecast_cache = None

    def __init__(self):
        self.base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        base_dir = Path(__file__).parent
        self.data_dir = base_dir / "dados" / "nomads"
        self.cache_dir = base_dir / "dados" / "cache"
        self.metadata_file = self.data_dir / "metadata.json"

        # Criar diretórios
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "dados" / "logs").mkdir(parents=True, exist_ok=True)

        # Configurações
        self.default_cycle = "06"  # Ciclo 06Z (03h BRT)
        self.forecast_hours = list(range(1, 168 ))  # f001 até f120
        self.max_retries = 3
        self.retry_delay = 60  # segundos

        # Região (bounding box) atualizada conforme solicitado
        self.region = {
            "leftlon": "-54.3",
            "rightlon": "-53.3",
            "toplat": "-29.0",
            "bottomlat": "-30.3"
        }

        # Variáveis meteorológicas
        self.variables = [
            "TMP",    # Temperatura
            "RH",     # Umidade relativa
            "APCP",   # Precipitação acumulada
            "UGRD",   # Componente U do vento
            "VGRD",   # Componente V do vento
            "DSWRF",  # Radiação solar descendente
            "TCDC",   # Cobertura de nuvens
            "PRES",   # Pressão
        ]

        # Níveis
        self.levels = [
            "2_m_above_ground",
            "10_m_above_ground",
            "surface"
        ]

        self.load_metadata()
        self._load_latest_forecast_to_cache()
    def _load_latest_forecast_to_cache(self):
        """Load latest consolidated forecast CSV into in-memory cache if exists."""
        try:
            forecast_files = list(self.cache_dir.glob("forecast_*.csv"))
            if forecast_files:
                latest_file = max(forecast_files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest_file)
                NOMADSDownloader._latest_forecast_cache = df
        except Exception as e:
            logger.warning(f"Could not load latest forecast to cache: {e}")
    
    def load_metadata(self):
        """Carrega metadados de downloads anteriores"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "last_update": None,
                "files": {},
                "statistics": {
                    "total_downloads": 0,
                    "total_size_mb": 0,
                    "failures": 0
                }
            }
    
    def save_metadata(self):
        """Salva metadados"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calcula hash do arquivo para verificar mudanças"""
        if not file_path.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def needs_update(self, date: datetime, cycle: str, forecast_hour: int) -> bool:
        """
        Verifica se o arquivo precisa ser atualizado
        """
        file_key = f"gfs_{date.strftime('%Y%m%d')}_{cycle}_f{forecast_hour:03d}"
        file_path = self.data_dir / f"{file_key}.grib2"
        
        # Se o arquivo não existe, precisa baixar
        if not file_path.exists():
            return True
        
        # Verificar se o arquivo mudou (comparar com metadata)
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.metadata.get("files", {}).get(file_key, {}).get("hash", "")
        
        if current_hash != stored_hash:
            return True
        
        # Verificar idade do arquivo (redownload após 7 dias)
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.days > 7:
            return True
        
        return False
    
    async def download_gfs_file(self, date: datetime, cycle: str, forecast_hour: int) -> Optional[Path]:
        """
        Baixa um arquivo GFS específico
        """
        date_str = date.strftime("%Y%m%d")
        hour_str = f"f{forecast_hour:03d}"
        file_name = f"gfs.t{cycle}z.pgrb2.0p25.{hour_str}"
        
        # Verificar se precisa atualizar
        if not self.needs_update(date, cycle, forecast_hour):
            logger.info(f"File {file_name} is up to date, skipping")
            return self.data_dir / f"gfs_{date_str}_{cycle}_{hour_str}.grib2"
        
        params = {
            "file": file_name,
            "dir": f"/gfs.{date_str}/{cycle}/atmos",
            "subregion": "",
            **self.region
        }
        
        # Adicionar variáveis
        for var in self.variables:
            params[f"var_{var}"] = "on"
        
        # Adicionar níveis
        for level in self.levels:
            params[f"lev_{level}"] = "on"
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=300)
                    async with session.get(self.base_url, params=params, timeout=timeout) as response:
                        if response.status == 200:
                            data = await response.read()
                            
                            # Salvar arquivo
                            output_file = self.data_dir / f"gfs_{date_str}_{cycle}_{hour_str}.grib2"
                            output_file.write_bytes(data)
                            
                            # Atualizar metadata
                            file_key = f"gfs_{date_str}_{cycle}_{hour_str}"
                            self.metadata["files"][file_key] = {
                                "hash": self.get_file_hash(output_file),
                                "size_mb": len(data) / (1024 * 1024),
                                "downloaded_at": datetime.now().isoformat(),
                                "date": date_str,
                                "cycle": cycle,
                                "forecast_hour": forecast_hour
                            }
                            
                            self.metadata["statistics"]["total_downloads"] += 1
                            self.metadata["statistics"]["total_size_mb"] += len(data) / (1024 * 1024)
                            
                            logger.info(f"Successfully downloaded {file_name} ({len(data)/1024/1024:.2f} MB)")
                            return output_file
                        else:
                            logger.warning(f"HTTP {response.status} for {file_name}")
                            
            except asyncio.TimeoutError:
                logger.error(f"Timeout downloading {file_name} (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Error downloading {file_name}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
        
        self.metadata["statistics"]["failures"] += 1
        logger.error(f"Failed to download {file_name} after {self.max_retries} attempts")
        return None
    
    async def process_grib_to_csv(self, grib_file: Path) -> Optional[Path]:
        """
        Processa arquivo GRIB e salva como CSV
        """
        try:
            # Abrir dataset
            ds = xr.open_dataset(grib_file, engine='cfgrib', 
                               backend_kwargs={'errors': 'ignore'})
            
            # Extrair informações do nome do arquivo
            parts = grib_file.stem.split('_')
            date_str = parts[1]
            cycle = parts[2]
            forecast_hour = int(parts[3][1:])
            
            # Calcular data/hora da previsão
            base_date = datetime.strptime(date_str, '%Y%m%d')
            base_date = base_date.replace(hour=int(cycle))
            forecast_date = base_date + timedelta(hours=forecast_hour)
            
            # Processar variáveis (média sobre a região)
            data = {
                'date': forecast_date.strftime('%Y-%m-%d'),
                'hour': forecast_date.hour,
                'forecast_base': base_date.isoformat(),
                'forecast_hour': forecast_hour
            }
            
            # Temperatura (converter K para C)
            if 't2m' in ds:
                temp = ds['t2m'].values - 273.15
                data['temp_media'] = float(np.nanmean(temp))
                data['temp_max'] = float(np.nanmax(temp))
                data['temp_min'] = float(np.nanmin(temp))
            
            # Umidade relativa
            if 'r2' in ds or 'rh' in ds:
                rh = ds['r2'].values if 'r2' in ds else ds['rh'].values
                data['umid_mediana'] = float(np.nanmedian(rh))
                data['umid_max'] = float(np.nanmax(rh))
                data['umid_min'] = float(np.nanmin(rh))
            
            # Precipitação
            if 'tp' in ds or 'acpcp' in ds:
                precip = ds['tp'].values if 'tp' in ds else ds['acpcp'].values
                data['precipitacao_total'] = float(np.nansum(precip))
            
            # Vento
            if 'u10' in ds and 'v10' in ds:
                u_wind = ds['u10'].values
                v_wind = ds['v10'].values
                wind_speed = np.sqrt(u_wind**2 + v_wind**2) * 3.6  # m/s para km/h
                data['vento_vel_media'] = float(np.nanmean(wind_speed))
                data['vento_raj_max'] = float(np.nanmax(wind_speed))
            
            # Radiação solar
            if 'dswrf' in ds:
                radiation = ds['dswrf'].values
                data['rad_mediana'] = float(np.nanmedian(radiation))
                data['rad_max'] = float(np.nanmax(radiation))
                data['rad_min'] = float(np.nanmin(radiation))
            
            # Salvar como CSV
            csv_file = self.cache_dir / f"{grib_file.stem}.csv"
            df = pd.DataFrame([data])
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Processed {grib_file.name} -> {csv_file.name}")
            return csv_file
            
        except Exception as e:
            logger.error(f"Error processing GRIB file {grib_file}: {e}")
            return None
    
    async def download_forecast_range(self, start_date: Optional[datetime] = None, days_ahead: int = 5):
        """
        Baixa previsões para os próximos dias
        """
        if start_date is None:
            start_date = datetime.now()
        
        logger.info(f"Starting download for {days_ahead} days from {start_date.strftime('%Y-%m-%d')}")
        
        # Determinar qual run do GFS usar (mais recente disponível)
        current_hour = datetime.now().hour
        if current_hour >= 9:  # Após 9h, usar run 06Z
            cycle = "06"
        elif current_hour >= 3:  # Após 3h, usar run 00Z
            cycle = "00"
        else:  # Usar run 18Z do dia anterior
            cycle = "18"
            start_date = start_date - timedelta(days=1)
        
        # Baixar arquivos
        tasks = []
        hours_to_download = list(range(1, min(days_ahead * 24 + 1, 168)))

        for hour in hours_to_download:
            task = self.download_gfs_file(start_date, cycle, hour)
            tasks.append(task)
        
        # Executar downloads em paralelo (com limite)
        semaphore = asyncio.Semaphore(3)  # Máximo 3 downloads simultâneos
        
        async def download_with_limit(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[download_with_limit(task) for task in tasks])
        
        # Processar arquivos baixados
        csv_files = []
        for grib_file in results:
            if grib_file:
                csv = await self.process_grib_to_csv(grib_file)
                if csv:
                    csv_files.append(csv)
        

        # Consolidar CSVs em um único arquivo
        if csv_files:
            all_data = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            consolidated = pd.concat(all_data, ignore_index=True)
            consolidated = consolidated.sort_values(['date', 'hour'])

            # Enriquecimento dos dados NOMADS para compatibilidade com o modelo
            consolidated['date'] = pd.to_datetime(consolidated['date'])
            consolidated['dia_semana_num'] = consolidated['date'].dt.weekday
            consolidated['day_of_year'] = consolidated['date'].dt.dayofyear
            consolidated['mes'] = consolidated['date'].dt.month
            consolidated['trimestre'] = consolidated['date'].dt.quarter

            # Marcar feriados nacionais do Brasil
            br_holidays = holidays.country_holidays('BR')
            consolidated['feriado'] = consolidated['date'].isin(br_holidays).astype(int)

            # Mapear variável de chuva compatível
            consolidated['Chuva_aberta'] = consolidated['precipitacao_total']

            # Adicionar loja_id para cada loja usada no modelo
            lojas = [1, 2]
            previsoes = []
            for loja in lojas:
                df_copy = consolidated.copy()
                df_copy['loja_id'] = loja
                previsoes.append(df_copy)
            df_final = pd.concat(previsoes, ignore_index=True)


            output_file = self.cache_dir / f"forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df_final.to_csv(output_file, index=False)
            logger.info(f"Consolidated forecast saved to {output_file}")
            # Limpa arquivos antigos
            self.cleanup_old_forecasts()

            # Update in-memory cache
            NOMADSDownloader._latest_forecast_cache = df_final
    @classmethod
    def get_latest_forecast_cache(cls):
        """Return the latest consolidated forecast from in-memory cache (DataFrame), or None."""
        return cls._latest_forecast_cache
        
        # Salvar metadata
        self.metadata["last_update"] = datetime.now().isoformat()
        self.save_metadata()
        
        logger.info("Download completed successfully")
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """
        Remove arquivos antigos para economizar espaço
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for file_path in self.data_dir.glob("*.grib2"):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                file_path.unlink()
                logger.info(f"Removed old file: {file_path.name}")
        
        for file_path in self.cache_dir.glob("*.csv"):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                file_path.unlink()
                logger.info(f"Removed old CSV: {file_path.name}")

async def main():
    """
    Função principal para execução manual ou teste
    """
    downloader = NOMADSDownloader()
    
    # Baixar previsões para os próximos 5 dias
    await downloader.download_forecast_range(days_ahead=5)
    
    # Limpar arquivos antigos
    downloader.cleanup_old_files(days_to_keep=30)

def scheduled_job():
    """
    Job para ser executado pelo scheduler
    """
    logger.info("Starting scheduled NOMADS update")
    asyncio.run(main())


def run_daily_job():
    """
    Executa o job uma vez por dia, sempre no mesmo horário (ex: 03:00).
    Se faz mais de 24h desde a última execução, executa imediatamente.
    """
    downloader = NOMADSDownloader()
    last_update = downloader.metadata.get("last_update")
    now = datetime.now()
    should_run_now = False
    if last_update:
        try:
            last_dt = datetime.fromisoformat(last_update)
            if (now - last_dt).total_seconds() > 23.5 * 3600:
                should_run_now = True
        except Exception as e:
            logger.warning(f"Erro ao ler data da última execução: {e}")
            should_run_now = True
    else:
        should_run_now = True

    def schedule_next_run():
        now2 = datetime.now()
        # Próxima execução às 03:00
        next_run = now2.replace(hour=3, minute=0, second=0, microsecond=0)
        if now2 >= next_run:
            next_run += timedelta(days=1)
        delay = (next_run - now2).total_seconds()
        logger.info(f"Próxima execução agendada para: {next_run}")
        threading.Timer(delay, run_and_reschedule).start()

    def run_and_reschedule():
        logger.info("Executando job diário NOMADS...")
        asyncio.run(main())
        schedule_next_run()

    if should_run_now:
        logger.info("Faz mais de 24h desde a última execução. Executando imediatamente...")
        asyncio.run(main())
    schedule_next_run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--scheduler":
        # Executar com novo scheduler para 6h/12h
        run_nomads_scheduled()
    elif len(sys.argv) > 1 and sys.argv[1] == "--daily":
        # Executar agendamento diário automático
        run_daily_job()
        while True:
            time.sleep(3600)  # Mantém o script rodando
    else:
        # Execução única
        asyncio.run(main())
