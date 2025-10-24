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
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "asterius.db")
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Banco SQLite não encontrado.")
        conn = sqlite3.connect(db_path)
        query = (
            "SELECT hour, temp_media, precipitacao_total "
            "FROM weather_hourly "
            "WHERE DATE(date) = ? AND loja_id = ? "
            "ORDER BY hour ASC"
        )
        cursor = conn.execute(query, (str(date), int(loja_id)))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            raise HTTPException(status_code=404, detail=f"Sem dados horários para {date}, loja {loja_id}.")
        data = []
        for row in rows:
            data.append({
                "hour": int(row[0]),
                "temperature": float(row[1]),
                "precipitation": float(row[2])
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
