from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import os
import glob
import pandas as pd

router = APIRouter()

@router.get("/hourly-weather/{date}/{loja_id}", tags=["Weather"])
def get_hourly_weather(date: str, loja_id: int = 1) -> Dict[str, Any]:
    """
    Retorna dados NOMADS horários (temperatura e precipitação) para um dia e loja específicos.
    Args:
        date: Data no formato YYYY-MM-DD
        loja_id: ID da loja
    Returns:
        Lista de dicts com hour, temp_media, precipitacao_total
    """
    try:
        # Try to get data from in-memory cache first
        from NOMADS.captar_nomads import NOMADSDownloader
        df = NOMADSDownloader.get_latest_forecast_cache()
        if df is not None:
            # Corrige tipos para evitar erro de filtro
            df['date'] = df['date'].astype(str).str.strip()
            df['loja_id'] = df['loja_id'].astype(str)
            df_filtered = df[(df['date'] == str(date).strip()) & (df['loja_id'] == str(loja_id))]
            if not df_filtered.empty:
                hourly = df_filtered[['hour', 'temp_media', 'precipitacao_total']].copy()
                hourly = hourly.sort_values('hour')
                data = []
                for _, row in hourly.iterrows():
                    data.append({
                        "hour": int(row['hour']),
                        "temperature": float(row['temp_media']),
                        "precipitation": float(row['precipitacao_total'])
                    })
                return {
                    "success": True,
                    "date": date,
                    "loja_id": loja_id,
                    "data": data
                }

        # Fallback to disk if cache is empty or no data for date/loja_id
        nomads_data_path = os.path.join(os.path.dirname(__file__), "NOMADS", "dados", "cache")
        forecast_files = glob.glob(os.path.join(nomads_data_path, "forecast_*.csv"))
        if not forecast_files:
            raise HTTPException(status_code=404, detail="Nenhum arquivo de forecast encontrado.")
        latest_file = max(forecast_files, key=os.path.getctime)
        df_disk = pd.read_csv(latest_file)
        df_disk['date'] = df_disk['date'].astype(str).str.strip()
        df_disk['loja_id'] = df_disk['loja_id'].astype(str)
        df_disk = df_disk[(df_disk['date'] == str(date).strip()) & (df_disk['loja_id'] == str(loja_id))]
        if df_disk.empty:
            raise HTTPException(status_code=404, detail=f"Sem dados para {date}, loja {loja_id}.")
        hourly = df_disk[['hour', 'temp_media', 'precipitacao_total']].copy()
        hourly = hourly.sort_values('hour')
        data = []
        for _, row in hourly.iterrows():
            data.append({
                "hour": int(row['hour']),
                "temperature": float(row['temp_media']),
                "precipitation": float(row['precipitacao_total'])
            })
        return {
            "success": True,
            "date": date,
            "loja_id": loja_id,
            "data": data
        }
    except Exception as e:
        print(f"[hourly-weather] Erro: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter dados horários: {str(e)}")
