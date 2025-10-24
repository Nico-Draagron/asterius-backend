"""
Script automatizado para baixar dados GFS do NOMADS - VERSÃO CORRIGIDA
Implementa lógica correta de precipitação acumulada seguindo o código R original
"""

import asyncio
import aiohttp
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import hashlib
from typing import Optional, List, Tuple
import warnings
import os
import holidays
import pytz

warnings.filterwarnings('ignore')

os.makedirs('NOMADS/dados/logs', exist_ok=True)

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
    def _sync_with_cloud_storage(self):
        """
        Sincroniza dados do Cloud Storage para o ambiente local se rodando no Cloud Run.
        """
        try:
            if os.getenv("K_SERVICE"):
                from backend import storage_manager
                sm = storage_manager.get_storage_manager()
                if sm:
                    # Sincronizar nomads/cache → NOMADS/dados/cache
                    sm.sync_gcs_to_local("nomads/cache", "NOMADS/dados/cache")
                    # Sincronizar modelo → modelo/
                    sm.sync_gcs_to_local("modelo", "modelo")
                    return sm
        except Exception as e:
            import logging
            logging.warning(f"[CloudStorage] Falha ao sincronizar do cloud: {e}")
        return None

    def _upload_to_cloud_storage(self, storage_manager):
        """
        Faz upload dos arquivos forecast_*.csv e logs para o Cloud Storage.
        """
        try:
            if storage_manager:
                # Upload dos forecasts
                from pathlib import Path
                cache_dir = Path("NOMADS/dados/cache")
                for file in cache_dir.glob("forecast_*.csv"):
                    storage_manager.upload_file(str(file), f"nomads/cache/{file.name}")
                # Upload dos logs
                logs_dir = Path("NOMADS/dados/logs")
                for file in logs_dir.glob("*.log"):
                    storage_manager.upload_file(str(file), f"nomads/logs/{file.name}")
        except Exception as e:
            import logging
            logging.warning(f"[CloudStorage] Falha ao fazer upload para cloud: {e}")
    """
    Classe para gerenciar downloads automatizados do NOMADS GFS
    COM LÓGICA CORRETA DE PRECIPITAÇÃO ACUMULADA
    """
    
    _latest_forecast_cache = None

    def __init__(self):
        self.base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        base_dir = Path(__file__).parent
        self.data_dir = base_dir / "dados" / "nomads"
        self.cache_dir = base_dir / "dados" / "cache"
        self.metadata_file = self.data_dir / "metadata.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "dados" / "logs").mkdir(parents=True, exist_ok=True)

        self.max_retries = 3
        self.retry_delay = 60

        # Região - Santa Maria, RS
        self.region = {
            "leftlon": "-54.3",
            "rightlon": "-53.3",
            "toplat": "-29.0",
            "bottomlat": "-30.3"
        }

        self.variables = [
            "TMP", "RH", "APCP", "UGRD", "VGRD", 
            "DSWRF", "TCDC", "PRES"
        ]

        self.levels = [
            "2_m_above_ground",
            "10_m_above_ground",
            "surface"
        ]

        self.load_metadata()
        self._load_latest_forecast_to_cache()

    def _load_latest_forecast_to_cache(self):
        try:
            forecast_files = list(self.cache_dir.glob("forecast_*.csv"))
            if forecast_files:
                latest_file = max(forecast_files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest_file)
                NOMADSDownloader._latest_forecast_cache = df
                logger.info(f"Loaded forecast cache from {latest_file.name}")
        except Exception as e:
            logger.warning(f"Could not load latest forecast to cache: {e}")
    
    def load_metadata(self):
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
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_file_hash(self, file_path: Path) -> str:
        if not file_path.exists():
            return ""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_latest_available_cycle(self) -> tuple:
        tz_br = pytz.timezone('America/Sao_Paulo')
        now_br = datetime.now(tz_br)
        now_utc = now_br.astimezone(pytz.UTC)
        
        check_time = now_utc - timedelta(hours=4)
        cycle_hour = (check_time.hour // 6) * 6
        cycle = f"{cycle_hour:02d}"
        cycle_date = check_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        
        logger.info(f"Using GFS cycle: {cycle_date.strftime('%Y%m%d')} {cycle}Z")
        logger.info(f"Current time UTC: {now_utc.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"Current time BRT: {now_br.strftime('%Y-%m-%d %H:%M')}")
        
        return cycle_date, cycle
    
    def needs_update(self, date: datetime, cycle: str, forecast_hour: int) -> bool:
        file_key = f"gfs_{date.strftime('%Y%m%d')}_{cycle}_f{forecast_hour:03d}"
        file_path = self.data_dir / f"{file_key}.grib2"
        
        if not file_path.exists():
            return True
        
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.metadata.get("files", {}).get(file_key, {}).get("hash", "")
        
        if current_hash != stored_hash:
            return True
        
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.days > 7:
            return True
        
        return False
    
    async def check_file_availability(self, date: datetime, cycle: str, forecast_hour: int) -> bool:
        date_str = date.strftime("%Y%m%d")
        hour_str = f"f{forecast_hour:03d}"
        file_name = f"gfs.t{cycle}z.pgrb2.0p25.{hour_str}"
        check_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date_str}/{cycle}/atmos/{file_name}"
        
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.head(check_url, timeout=timeout, allow_redirects=True) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"File {file_name} not available: {e}")
            return False
    
    async def download_gfs_file(self, date: datetime, cycle: str, forecast_hour: int) -> Optional[Path]:
        date_str = date.strftime("%Y%m%d")
        hour_str = f"f{forecast_hour:03d}"
        file_name = f"gfs.t{cycle}z.pgrb2.0p25.{hour_str}"
        
        output_file = self.data_dir / f"gfs_{date_str}_{cycle}_{hour_str}.grib2"
        if not self.needs_update(date, cycle, forecast_hour):
            if output_file.exists():
                logger.info(f"File {file_name} is up to date, skipping")
                return output_file
        
        is_available = await self.check_file_availability(date, cycle, forecast_hour)
        if not is_available:
            logger.warning(f"File {file_name} not yet available on NOMADS server")
            return None
        
        params = {
            "file": file_name,
            "dir": f"/gfs.{date_str}/{cycle}/atmos",
            "subregion": "",
            **self.region
        }
        
        for var in self.variables:
            params[f"var_{var}"] = "on"
        for level in self.levels:
            params[f"lev_{level}"] = "on"
        
        for attempt in range(self.max_retries):
            try:
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=3)
                timeout = aiohttp.ClientTimeout(total=300, connect=60)
                
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        self.base_url, 
                        params=params, 
                        timeout=timeout,
                        allow_redirects=True,
                        max_redirects=10
                    ) as response:
                        if response.status in [302, 301]:
                            logger.warning(f"Redirect detected for {file_name}")
                            await asyncio.sleep(30)
                            continue
                        
                        if response.status == 200:
                            data = await response.read()
                            
                            if len(data) < 100 or not data[:4] == b'GRIB':
                                logger.error(f"Invalid GRIB file for {file_name}")
                                if attempt < self.max_retries - 1:
                                    await asyncio.sleep(self.retry_delay)
                                    continue
                                return None
                            
                            output_file.write_bytes(data)
                            
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
                            
                            logger.info(f"Downloaded {file_name} ({len(data)/1024/1024:.2f} MB)")
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
        return None

    def extract_data_from_grib(self, grib_file: Path) -> Optional[dict]:
        """
        Extrai dados de um arquivo GRIB individual
        Retorna dicionário com os valores
        """
        try:
            ds = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
            
            lon = (float(self.region['leftlon']) + float(self.region['rightlon'])) / 2
            lat = (float(self.region['toplat']) + float(self.region['bottomlat'])) / 2
            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
            
            parts = grib_file.stem.split('_')
            forecast_hour = int(parts[3][1:])
            
            # Extrair valores escalares
            data = {'forecast_hour': forecast_hour}
            
            # Precipitação ACUMULADA desde início do ciclo
            if 'tp' in ds:
                data['precip_acum'] = float(ds['tp'].values)
            elif 'acpcp' in ds:
                data['precip_acum'] = float(ds['acpcp'].values)
            else:
                data['precip_acum'] = 0.0
            
            # Radiação ACUMULADA
            if 'dswrf' in ds:
                data['rad_acum'] = float(ds['dswrf'].values)
            else:
                data['rad_acum'] = 0.0
            
            # Temperatura (instantânea)
            if 't2m' in ds:
                data['temp'] = float(ds['t2m'].values) - 273.15
            else:
                data['temp'] = None
            
            # Umidade (instantânea)
            if 'r2' in ds:
                data['rh'] = float(ds['r2'].values)
            elif 'rh' in ds:
                data['rh'] = float(ds['rh'].values)
            else:
                data['rh'] = None
            
            # Vento (instantâneo)
            if 'u10' in ds and 'v10' in ds:
                u = float(ds['u10'].values)
                v = float(ds['v10'].values)
                data['wind_speed'] = float(np.sqrt(u**2 + v**2) * 3.6)  # m/s para km/h
            else:
                data['wind_speed'] = None
            
            ds.close()
            return data
            
        except Exception as e:
            logger.error(f"Error extracting data from {grib_file}: {e}")
            return None

    def process_accumulated_to_hourly(self, data_list: List[dict], cycle_date: datetime) -> pd.DataFrame:
        """
        Processa dados acumulados para valores horários
        Implementa EXATAMENTE a lógica do código R
        
        Lógica R:
        PRECx = PREC_SM  # valores acumulados
        k = 1
        for (j in 1:(4*nd)) {        # para cada bloco de 6h
            for (i in 1:6) {         # para cada hora do bloco
                k = k + 1
                if (i != 1) {
                    PRECx[k,] = PREC_SM[k,] - PREC_SM[k-1,]
                }
            }
        }
        """
        tz_br = pytz.timezone('America/Sao_Paulo')
        # Corrige erro: só localiza se datetime for naive
        if cycle_date.tzinfo is None:
            cycle_date_utc = pytz.UTC.localize(cycle_date)
        else:
            cycle_date_utc = cycle_date.astimezone(pytz.UTC)
        base_date_local = cycle_date_utc.astimezone(tz_br)
        
        # Ordenar por forecast_hour
        data_list = sorted(data_list, key=lambda x: x['forecast_hour'])
        
        rows = []
        bloco_size = 6  # blocos de 6 horas
        
        # Processar em blocos de 6 horas
        num_blocos = len(data_list) // bloco_size
        
        logger.info(f"Processing {len(data_list)} forecast hours in {num_blocos} blocks of 6h")
        
        for j in range(num_blocos):
            # Índices do bloco atual
            bloco_start_idx = j * bloco_size
            bloco_end_idx = min(bloco_start_idx + bloco_size, len(data_list))
            
            if bloco_end_idx <= bloco_start_idx:
                break
            
            # Pegar dados acumulados no FINAL do bloco
            data_fim_bloco = data_list[bloco_end_idx - 1]
            
            # Pegar dados acumulados no FINAL do bloco ANTERIOR (se existir)
            if j > 0:
                data_fim_bloco_anterior = data_list[bloco_start_idx - 1]
                # Precipitação INCREMENTAL deste bloco
                precip_bloco = data_fim_bloco['precip_acum'] - data_fim_bloco_anterior['precip_acum']
                rad_bloco = data_fim_bloco['rad_acum'] - data_fim_bloco_anterior['rad_acum']
            else:
                # Primeiro bloco: usar valor acumulado diretamente
                precip_bloco = data_fim_bloco['precip_acum']
                rad_bloco = data_fim_bloco['rad_acum']
            
            # Distribuir uniformemente nas 6 horas
            precip_horaria = precip_bloco / bloco_size
            rad_horaria = rad_bloco / bloco_size
            
            # Criar linhas para cada hora do bloco
            for i in range(bloco_size):
                idx = bloco_start_idx + i
                if idx >= len(data_list):
                    break
                
                data_hora = data_list[idx]
                hora_local = base_date_local + timedelta(hours=data_hora['forecast_hour'])
                
                row = {
                    'date': hora_local.strftime('%Y-%m-%d'),
                    'hour': hora_local.hour,
                    'forecast_base': base_date_local.isoformat(),
                    'forecast_hour': data_hora['forecast_hour'],
                    'precipitacao_total': precip_horaria,
                    'rad_total': rad_horaria,
                    'temp_media': data_hora['temp'],
                    'temp_max': data_hora['temp'],
                    'temp_min': data_hora['temp'],
                    'umid_mediana': data_hora['rh'],
                    'umid_max': data_hora['rh'],
                    'umid_min': data_hora['rh'],
                    'vento_vel_media': data_hora['wind_speed'],
                    'vento_raj_max': data_hora['wind_speed'],
                    'rad_mediana': rad_horaria,
                    'rad_max': rad_horaria,
                    'rad_min': rad_horaria
                }
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} hourly records from accumulated data")
        
        return df

    async def download_forecast_range(self, start_date: Optional[datetime] = None, days_ahead: int = 5):
    # Integração Cloud Storage: baixar dados do bucket se rodando no Cloud Run
    storage_manager = self._sync_with_cloud_storage()
        """
        Baixa previsões e processa com lógica correta de acumulados
        """
        cycle_date, cycle = self.get_latest_available_cycle()
        
        logger.info(f"Starting download for {days_ahead} days from {cycle_date.strftime('%Y-%m-%d')} cycle {cycle}Z")
        
        max_forecast_hours = min(days_ahead * 24, 120)
        hours_to_download = list(range(1, max_forecast_hours + 1))
        
        logger.info(f"Will download {len(hours_to_download)} forecast hours")
        
        # Download em lotes
        batch_size = 10
        all_grib_files = []
        
        for i in range(0, len(hours_to_download), batch_size):
            batch = hours_to_download[i:i + batch_size]
            logger.info(f"Downloading batch {i//batch_size + 1}/{(len(hours_to_download)-1)//batch_size + 1}")
            
            tasks = [self.download_gfs_file(cycle_date, cycle, hour) for hour in batch]
            semaphore = asyncio.Semaphore(3)
            
            async def download_with_limit(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(*[download_with_limit(task) for task in tasks])
            valid_files = [f for f in results if f is not None]
            all_grib_files.extend(valid_files)
            
            logger.info(f"Batch completed: {len(valid_files)}/{len(batch)} files")
            
            if i + batch_size < len(hours_to_download):
                await asyncio.sleep(2)
        
        logger.info(f"Total files downloaded: {len(all_grib_files)}/{len(hours_to_download)}")
        
        # NOVA LÓGICA: Extrair dados de todos os arquivos primeiro
        all_data = []
        for grib_file in all_grib_files:
            data = self.extract_data_from_grib(grib_file)
            if data:
                all_data.append(data)
        
        logger.info(f"Extracted data from {len(all_data)} GRIB files")
        
        if not all_data:
            logger.error("No data extracted from GRIB files")
            return
        


        # Gerar DataFrame horário a partir dos dados extraídos
        df_hourly = self.process_accumulated_to_hourly(all_data, cycle_date)
        if df_hourly.empty:
            logger.error("No hourly data generated")
            return

        # Enriquecimento
        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
        df_hourly['dia_semana_num'] = df_hourly['date'].dt.weekday
        df_hourly['day_of_year'] = df_hourly['date'].dt.dayofyear
        df_hourly['mes'] = df_hourly['date'].dt.month
        df_hourly['trimestre'] = df_hourly['date'].dt.quarter

        br_holidays = holidays.country_holidays('BR')
        df_hourly['feriado'] = df_hourly['date'].apply(lambda x: int(x in br_holidays))
        df_hourly['Chuva_aberta'] = df_hourly['precipitacao_total']

        # Adicionar lojas
        lojas = [1, 2]
        previsoes = []
        for loja in lojas:
            df_copy = df_hourly.copy()
            df_copy['loja_id'] = loja
            previsoes.append(df_copy)
        df_final = pd.concat(previsoes, ignore_index=True)

        # Salvar CSV detalhado hora a hora

        # Salvar dados horários no SQLite
        try:
            from database import AsteriusDB
            db = AsteriusDB()
            db.import_hourly_weather(df_final)
            logger.info(f"Hourly forecast saved to SQLite (weather_hourly)")
        except Exception as e:
            logger.error(f"Erro ao salvar dados horários no SQLite: {e}")


        # Agrupar por data (somar precipitação, agregar outras)
        agg_dict = {
            'temp_max': 'max',
            'temp_min': 'min',
            'temp_media': 'mean',
            'umid_mediana': 'mean',
            'umid_max': 'max',
            'umid_min': 'min',
            'precipitacao_total': 'sum',  # IMPORTANTE: SOMAR não fazer média!
            'Chuva_aberta': 'sum',
            'rad_total': 'sum',
            'rad_mediana': 'mean',
            'rad_max': 'max',
            'rad_min': 'min',
            'vento_vel_media': 'mean',
            'vento_raj_max': 'max',
            'dia_semana_num': 'first',
            'day_of_year': 'first',
            'mes': 'first',
            'trimestre': 'first',
            'feriado': 'first',
            'forecast_base': 'first'
        }

        # Garantir que todas as colunas do agg_dict existam no DataFrame
        for col in agg_dict.keys():
            if col not in df_final.columns:
                df_final[col] = np.nan

        df_grouped = df_final.groupby(['date', 'loja_id'], as_index=False).agg(agg_dict)

        output_file = self.cache_dir / f"forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df_grouped.to_csv(output_file, index=False)
        logger.info(f"Consolidated forecast saved to {output_file}")
        
        # Limpar arquivos antigos
        self.cleanup_old_forecasts()
        
        # Atualizar cache
        NOMADSDownloader._latest_forecast_cache = df_grouped
        logger.info(f"Forecast cache updated with {len(df_grouped)} records")
        
        # Salvar metadata
    # Upload para Cloud Storage (se disponível)
    self._upload_to_cloud_storage(storage_manager)
    self.metadata["last_update"] = datetime.now().isoformat()
    self.save_metadata()
        
    logger.info("Download completed successfully")
    
    def cleanup_old_forecasts(self):
        # Remover apenas arquivos diários, não os horários
        daily_files = [f for f in self.cache_dir.glob("forecast_*.csv") if not f.name.startswith("forecast_hourly_")]
        if len(daily_files) <= 1:
            return
        latest_file = max(daily_files, key=lambda f: f.stat().st_mtime)
        for f in daily_files:
            if f != latest_file:
                try:
                    f.unlink()
                    logger.info(f"Removed old forecast: {f.name}")
                except Exception as e:
                    logger.warning(f"Não foi possível remover {f}: {e}")
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        removed_count = 0
        for file_path in self.data_dir.glob("*.grib2"):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                file_path.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old GRIB2 files")
    
    @classmethod
    def get_latest_forecast_cache(cls):
        return cls._latest_forecast_cache


async def main():
    downloader = NOMADSDownloader()
    await downloader.download_forecast_range(days_ahead=5)
    downloader.cleanup_old_files(days_to_keep=30)


if __name__ == "__main__":
    import sys
    import time
    import threading
    
    def scheduled_job():
        logger.info("Starting scheduled NOMADS update")
        asyncio.run(main())
    
    def run_nomads_scheduled():
        def job():
            logger.info("[Scheduler] Running NOMADSDownloader job...")
            downloader = NOMADSDownloader()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(downloader.download_forecast_range(days_ahead=5))
            downloader.cleanup_old_files(days_to_keep=30)
            logger.info("[Scheduler] Job completed.")

        def get_seconds_until(hour, minute):
            tz = pytz.timezone('America/Sao_Paulo')
            now = datetime.now(tz)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            return (target - now).total_seconds()

        def schedule_next_run():
            seconds_until_6am = get_seconds_until(6, 0)
            seconds_until_12pm = get_seconds_until(12, 0)
            next_run = min(seconds_until_6am, seconds_until_12pm)
            next_time_str = "6:00" if next_run == seconds_until_6am else "12:00"
            logger.info(f"[Scheduler] Next run in {next_run/3600:.2f}h at {next_time_str} BRT")
            timer = threading.Timer(next_run, run_scheduled)
            timer.daemon = True
            timer.start()
            return timer

        def run_scheduled():
            try:
                job()
            except Exception as e:
                logger.error(f"[Scheduler] Error: {e}", exc_info=True)
            finally:
                schedule_next_run()

        logger.info("[Scheduler] Starting (runs at 6:00 and 12:00 BRT)")
        return schedule_next_run()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--scheduled":
        logger.info("Starting in scheduled mode...")
        timer = run_nomads_scheduled()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped")
            timer.cancel()
    else:
        logger.info("Starting single execution...")
        asyncio.run(main())
        logger.info("Execution completed")