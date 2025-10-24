import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime, timedelta
import glob

logger = logging.getLogger(__name__)

class XGBoostModel:
    """Wrapper para modelo XGBoost com dados reais do NOMADS"""
    def __init__(self, model_path: str = None):
        # Usa variável de ambiente ML_MODEL_PATH, se não for fornecido explicitamente
        from pathlib import Path
        if model_path is None:
            # Caminho absoluto baseado na raiz do projeto (asterius)
            project_root = Path(__file__).resolve().parent
            # Sobe até encontrar a pasta backend
            while project_root.name != "backend" and project_root.parent != project_root:
                project_root = project_root.parent
            model_path = os.getenv("ML_MODEL_PATH", str(project_root / "modelo" / "modelo_final_xgb.pkl"))
        self.model_path = model_path
        self.model = None
        self.is_mock = False
        # Ensure path is anchored to the backend module location (avoid CWD issues)
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent
        self.nomads_data_path = str(base_dir / "NOMADS" / "dados" / "cache")
        self._load_model()

    def get_all_weather_days(self, loja_id: int = 1) -> List[Dict[str, Any]]:
        """Retorna os dados meteorológicos de todas as datas disponíveis no banco SQLite para a loja informada."""
        import sqlite3
        db_path = os.path.join(self.nomads_data_path, "nomads_forecast.db")
        if not os.path.exists(db_path):
            logger.warning(f"Banco SQLite não encontrado: {db_path}")
            return []
        try:
            conn = sqlite3.connect(db_path)
            query = (
                "SELECT date, MAX(temp_max) as temp_max, AVG(temp_media) as temp_media, "
                "umid_mediana, day_of_year, Chuva_aberta, dia_semana_num, feriado, trimestre, mes, loja_id, precipitacao_total "
                "FROM forecast "
                "WHERE loja_id = ? "
                "GROUP BY date "
                "ORDER BY date ASC"
            )
            cursor = conn.execute(query, (loja_id,))
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    'date': str(row[0]),
                    'temp_max': float(row[1]),
                    'temperature': float(row[1]),
                    'temp_media': float(row[2]),
                    'umid_mediana': float(row[3]),
                    'rad_max': 800.0,
                    'day_of_year': int(row[4]),
                    'Chuva_aberta': self._normalize_precipitation(row[5]),
                    'dia_semana_num': int(row[6]),
                    'feriado': int(row[7]),
                    'trimestre': int(row[8]),
                    'mes': int(row[9]),
                    'loja_id': int(row[10]),
                    'precipitacao_total': self._normalize_precipitation(row[11])
                })
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Erro ao carregar todos os dados NOMADS do SQLite: {e}")
            return []
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime, timedelta
import glob

logger = logging.getLogger(__name__)

class XGBoostModel:
    """Wrapper para modelo XGBoost com dados reais do NOMADS"""
    
    def __init__(self, model_path: str = None):
        # Usa variável de ambiente ML_MODEL_PATH, se não for fornecido explicitamente
        from pathlib import Path
        if model_path is None:
            # Caminho absoluto baseado na raiz do projeto (asterius)
            project_root = Path(__file__).resolve().parent
            # Sobe até encontrar a pasta backend
            while project_root.name != "backend" and project_root.parent != project_root:
                project_root = project_root.parent
            model_path = os.getenv("ML_MODEL_PATH", str(project_root / "modelo" / "modelo_final_xgb.pkl"))
        self.model_path = model_path
        self.model = None
        self.is_mock = False
        # Ensure path is anchored to the backend module location (avoid CWD issues)
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent
        self.nomads_data_path = str(base_dir / "NOMADS" / "dados" / "cache")
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo ou usa fallback mock"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Modelo XGBoost carregado: {os.path.basename(self.model_path)}")
                self.is_mock = False
            else:
                logger.warning(f"❌ Modelo não encontrado em {self.model_path}. Usando mock.")
                self.is_mock = True
        except Exception as e:
            logger.error(f"💥 Erro ao carregar modelo: {e}. Usando mock.")
            self.is_mock = True

    def _normalize_precipitation(self, precip_value: float) -> float:
        """
        Normaliza valores de precipitação dos dados NOMADS para escala real (mm)
        Os dados NOMADS parecem vir em escala incorreta
        """
        try:
            value = float(precip_value)
            
            # Se o valor é muito alto, aplica conversões progressivas
            if value > 1000:
                # Primeira tentativa: dividir por 1000 (kg/m² para mm)
                value = value / 1000
            
            if value > 500:
                # Segunda tentativa: dividir por 100 adicional
                value = value / 100
            
            if value > 200:
                # Terceira tentativa: dividir por 10 adicional
                value = value / 10
            
            # Limita a valores realistas (0-100mm por período)
            value = max(0.0, min(value, 100.0))
            
            # Se ainda for muito alto, usa uma conversão logarítmica
            if value > 50:
                value = 50 * (1 - np.exp(-value/50))
            
            return round(value, 2)
            
        except (ValueError, TypeError):
            return 0.0

    def _get_default_weather_data(self) -> Dict[str, Any]:
        """Retorna dados meteorológicos padrão quando não há dados NOMADS"""
        return {
            'temp_max': 25.0,
            'temp_media': 22.0,
            'umid_mediana': 65.0,
            'rad_max': 800.0,
            'day_of_year': 280,
            'Chuva_aberta': 0.0,
            'dia_semana_num': 1,
            'feriado': 0,
            'trimestre': 4,
            'mes': 10,
            'loja_id': 1,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'precipitacao_total': 0.0
        }
    def get_weather_data(self, target_date: Optional[str] = None, loja_id: int = 1) -> Dict[str, Any]:
        """Obtém dados meteorológicos do banco SQLite para uma data específica"""
        import sqlite3
        db_path = os.path.join(self.nomads_data_path, "nomads_forecast.db")
        if not os.path.exists(db_path):
            logger.warning(f"Banco SQLite não encontrado: {db_path}")
            return self._get_default_weather_data()
        try:
            conn = sqlite3.connect(db_path)
            query = (
                "SELECT date, MAX(temp_max) as temp_max, AVG(temp_media) as temp_media, "
                "umid_mediana, day_of_year, Chuva_aberta, dia_semana_num, feriado, trimestre, mes, loja_id, precipitacao_total "
                "FROM forecast "
                f"WHERE loja_id = ? {date_filter} "
                "GROUP BY date "
                "ORDER BY date ASC "
                "LIMIT 1"
            )
            date_filter = ""
            params = [loja_id]
            if target_date:
                date_filter = "AND date = ?"
                params.append(target_date)
            query = query.format(date_filter=date_filter)
            cursor = conn.execute(query, tuple(params))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return self._get_default_weather_data()
            return {
                'temp_max': float(row[1]),
                'temperature': float(row[1]),
                'temp_media': float(row[2]),
                'umid_mediana': float(row[3]),
                'rad_max': 800.0,
                'day_of_year': int(row[4]),
                'Chuva_aberta': self._normalize_precipitation(row[5]),
                'dia_semana_num': int(row[6]),
                'feriado': int(row[7]),
                'trimestre': int(row[8]),
                'mes': int(row[9]),
                'loja_id': int(row[10]),
                'date': str(row[0]),
                'precipitacao_total': self._normalize_precipitation(row[11])
            }
        except Exception as e:
            logger.error(f"Erro ao carregar dados NOMADS do SQLite: {e}")
            return self._get_default_weather_data()
    
    def get_weather_forecast(self, days_ahead: int = 5, loja_id: int = 1) -> List[Dict[str, Any]]:
        """Obtém previsão meteorológica para vários dias do banco SQLite"""
        import sqlite3
        db_path = os.path.join(self.nomads_data_path, "nomads_forecast.db")
        if not os.path.exists(db_path):
            return [self._get_default_weather_data() for _ in range(days_ahead)]
        try:
            conn = sqlite3.connect(db_path)
            query = (
                "SELECT date, MAX(temp_max) as temp_max, AVG(temp_media) as temp_media, "
                "umid_mediana, day_of_year, Chuva_aberta, dia_semana_num, feriado, trimestre, mes, loja_id, precipitacao_total "
                "FROM forecast "
                "WHERE loja_id = ? "
                "GROUP BY date "
                "ORDER BY date ASC "
                "LIMIT ?"
            )
            cursor = conn.execute(query, (loja_id, days_ahead))
            rows = cursor.fetchall()
            forecast = []
            for row in rows:
                forecast.append({
                    'date': str(row[0]),
                    'temp_max': float(row[1]),
                    'temp_media': float(row[2]),
                    'umid_mediana': float(row[3]),
                    'precipitacao_total': self._normalize_precipitation(row[11]),
                    'Chuva_aberta': self._normalize_precipitation(row[5]),
                    'hour': 12  # SQLite não armazena hora, usa valor padrão
                })
            conn.close()
            return forecast if forecast else [self._get_default_weather_data() for _ in range(days_ahead)]
        except Exception as e:
            logger.error(f"Erro ao obter previsão do SQLite: {e}")
            return [self._get_default_weather_data() for _ in range(days_ahead)]
    
    def predict(self, target_date: Optional[str] = None, loja_id: int = 1) -> Dict[str, Any]:
        """Faz predição usando modelo real com dados NOMADS"""
        try:
            # Obtém dados meteorológicos reais
            weather_data = self.get_weather_data(target_date, loja_id)
            
            if self.is_mock or self.model is None:
                return self._mock_prediction(weather_data)
            
            # Preparar dados para o modelo real
            # O modelo espera um DataFrame com as features corretas
            feature_df = pd.DataFrame([{
                'temp_max': weather_data['temp_max'],
                'temp_media': weather_data['temp_media'],
                'umid_mediana': weather_data['umid_mediana'],
                'rad_max': weather_data['rad_max'],
                'day_of_year': weather_data['day_of_year'], 
                'Chuva_aberta': weather_data['Chuva_aberta'],
                'dia_semana_num': weather_data['dia_semana_num'],
                'feriado': weather_data['feriado'],
                'trimestre': weather_data['trimestre'],
                'mes': weather_data['mes'],
                'loja_id': weather_data['loja_id']
            }])
            
            # Faz a predição
            prediction = self.model.predict(feature_df)[0]
            
            return {
                "prediction": float(prediction),
                "confidence": 0.85,
                "model_type": "xgboost_real",
                "weather_data": weather_data,
                "target_date": weather_data['date'],
                "loja_id": loja_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            weather_data = self._get_default_weather_data()
            return self._mock_prediction(weather_data)
    
    def predict_multiple_days(self, days_ahead: int = 5, loja_id: int = 1) -> List[Dict[str, Any]]:
        """Faz predições para múltiplos dias"""
        try:
            forecast_data = self.get_weather_forecast(days_ahead, loja_id)
            predictions = []
            
            for day_data in forecast_data:
                if self.is_mock or self.model is None:
                    pred = self._mock_prediction(day_data)
                else:
                    # Preparar features para o modelo
                    feature_df = pd.DataFrame([{
                        'temp_max': day_data.get('temp_max', 25.0),
                        'temp_media': day_data.get('temp_media', 22.0),
                        'umid_mediana': day_data.get('umid_mediana', 65.0),
                        'rad_max': 800.0,
                        'day_of_year': pd.to_datetime(day_data['date']).dayofyear,
                        'Chuva_aberta': day_data.get('Chuva_aberta', 0.0),
                        'dia_semana_num': pd.to_datetime(day_data['date']).weekday(),
                        'feriado': 0,  # Simplificado
                        'trimestre': (pd.to_datetime(day_data['date']).month - 1) // 3 + 1,
                        'mes': pd.to_datetime(day_data['date']).month,
                        'loja_id': loja_id
                    }])
                    
                    prediction_value = self.model.predict(feature_df)[0]
                    pred = {
                        "prediction": float(prediction_value),
                        "confidence": 0.85,
                        "model_type": "xgboost_real",
                        "date": day_data['date'],
                        "weather_data": day_data
                    }
                
                predictions.append(pred)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erro na predição múltipla: {e}")
            return []
    
    def predict_hypothetical_scenario(self, 
                                    target_date: Optional[str] = None, 
                                    loja_id: int = 1,
                                    modifications: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Faz predição com cenários hipotéticos modificando variáveis específicas
        
        Args:
            target_date: Data alvo (None = hoje)
            loja_id: ID da loja
            modifications: Dicionário com modificações nas variáveis
                          Ex: {'Chuva_aberta': 0.0, 'temp_max': 30.0}
        
        Returns:
            Dict com predição original, predição hipotética e diferença
        """
        try:
            # Obtém dados meteorológicos reais
            weather_data = self.get_weather_data(target_date, loja_id)
            
            # Predição com dados reais (cenário atual)
            original_prediction = self.predict(target_date, loja_id)
            
            if modifications is None:
                modifications = {}
            
            # Aplica modificações aos dados meteorológicos
            modified_weather = weather_data.copy()
            for key, value in modifications.items():
                if key in modified_weather:
                    modified_weather[key] = value
            
            # Faz predição com dados modificados
            if self.is_mock or self.model is None:
                hypothetical_prediction = self._mock_prediction(modified_weather)
            else:
                # Preparar dados para o modelo real
                feature_df = pd.DataFrame([{
                    'temp_max': modified_weather['temp_max'],
                    'temp_media': modified_weather['temp_media'],
                    'umid_mediana': modified_weather['umid_mediana'],
                    'rad_max': modified_weather['rad_max'],
                    'day_of_year': modified_weather['day_of_year'],
                    'Chuva_aberta': modified_weather['Chuva_aberta'],
                    'dia_semana_num': modified_weather['dia_semana_num'],
                    'feriado': modified_weather['feriado'],
                    'trimestre': modified_weather['trimestre'],
                    'mes': modified_weather['mes'],
                    'loja_id': modified_weather['loja_id']
                }])
                
                prediction_value = self.model.predict(feature_df)[0]
                hypothetical_prediction = {
                    "prediction": float(prediction_value),
                    "confidence": 0.85,
                    "model_type": "xgboost_hypothetical",
                    "weather_data": modified_weather,
                    "target_date": modified_weather['date'],
                    "loja_id": loja_id,
                    "timestamp": datetime.now().isoformat(),
                    "modifications": modifications
                }
            
            # Calcula diferença
            difference = hypothetical_prediction['prediction'] - original_prediction['prediction']
            percentage_change = (difference / original_prediction['prediction']) * 100 if original_prediction['prediction'] != 0 else 0
            
            return {
                "original_scenario": original_prediction,
                "hypothetical_scenario": hypothetical_prediction,
                "comparison": {
                    "difference": float(difference),
                    "percentage_change": float(percentage_change),
                    "impact": "positive" if difference > 0 else "negative" if difference < 0 else "neutral"
                },
                "modifications_applied": modifications,
                "scenario_analysis": self._analyze_scenario_impact(modifications, difference, percentage_change)
            }
            
        except Exception as e:
            logger.error(f"Erro na predição hipotética: {e}")
            return {
                "error": str(e),
                "original_scenario": None,
                "hypothetical_scenario": None
            }
    
    def _analyze_scenario_impact(self, modifications: Dict[str, Any], difference: float, percentage_change: float) -> str:
        """Analisa o impacto das modificações no cenário"""
        analysis = []
        
        # Analisa cada modificação
        for key, value in modifications.items():
            if key == 'Chuva_aberta':
                if value == 0.0:
                    analysis.append("🌞 Cenário sem chuva")
                elif value < 5.0:
                    analysis.append("🌦️ Cenário com chuva leve")
                elif value < 20.0:
                    analysis.append("🌧️ Cenário com chuva moderada")
                else:
                    analysis.append("⛈️ Cenário com chuva forte")
            
            elif key == 'temp_max':
                if value > 30:
                    analysis.append(f"🔥 Temperatura alta ({value}°C)")
                elif value < 15:
                    analysis.append(f"❄️ Temperatura baixa ({value}°C)")
                else:
                    analysis.append(f"🌡️ Temperatura moderada ({value}°C)")
            
            elif key == 'umid_mediana':
                if value > 80:
                    analysis.append(f"💧 Alta umidade ({value}%)")
                elif value < 40:
                    analysis.append(f"🏜️ Baixa umidade ({value}%)")
        
        # Adiciona impacto financeiro
        if abs(difference) > 100:
            impact_text = f"Impacto significativo: R$ {difference:+,.2f} ({percentage_change:+.1f}%)"
        elif abs(difference) > 50:
            impact_text = f"Impacto moderado: R$ {difference:+,.2f} ({percentage_change:+.1f}%)"
        else:
            impact_text = f"Impacto pequeno: R$ {difference:+,.2f} ({percentage_change:+.1f}%)"
        
        analysis.append(impact_text)
        
        return " | ".join(analysis)
    
    def _mock_prediction(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predição mock baseada em dados meteorológicos"""
        base_value = 2500.0
        
        # Ajustes baseados em dados reais
        temp_effect = (weather_data.get('temp_max', 25) - 25) * 50
        base_value += temp_effect
        
        humid_effect = (70 - weather_data.get('umid_mediana', 65)) * 10
        base_value += humid_effect
        
        rain_effect = weather_data.get('Chuva_aberta', 0) * -0.1  # Chuva reduz vendas
        base_value += rain_effect
        
        # Variação aleatória
        variation = np.random.normal(0, 200)
        prediction = max(0, base_value + variation)
        
        return {
            "prediction": float(prediction),
            "confidence": 0.75,
            "model_type": "mock_with_real_weather",
            "weather_data": weather_data,
            "target_date": weather_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            "timestamp": datetime.now().isoformat(),
            "note": "Predição simulada usando dados meteorológicos reais"
        }


# Classe de compatibilidade para manter o código existente funcionando
class SalesPredictor:
    """Classe de compatibilidade que usa XGBoostModel"""
    
    def __init__(self, model_path: str = None):
        self.xgb_model = XGBoostModel(model_path)
        self.is_loaded = not self.xgb_model.is_mock
    
    def predict(self, days: int = 7, loja_id: int = 1) -> List[float]:
        """Compatibilidade com a API antiga"""
        predictions = self.xgb_model.predict_multiple_days(days, loja_id)
        return [pred['prediction'] for pred in predictions]
    
    def get_statistics(self, predictions: List[float]) -> dict:
        """Compatibilidade com a API antiga"""
        if not predictions:
            return {}
        
        return {
            "mean": round(np.mean(predictions), 2),
            "median": round(np.median(predictions), 2),
            "std": round(np.std(predictions), 2),
            "min": round(min(predictions), 2),
            "max": round(max(predictions), 2),
            "total": round(sum(predictions), 2)
        }


# Funções de conveniência
def predict_sales(days: int = 7, loja_id: int = 1, model_path: str = None) -> List[float]:
    """Função de conveniência para prever vendas com dados reais"""
    model = XGBoostModel(model_path)
    predictions = model.predict_multiple_days(days, loja_id)
    return [pred['prediction'] for pred in predictions]


def get_prediction_summary(days: int = 7, loja_id: int = 1) -> dict:
    """Retorna um resumo completo da previsão com dados meteorológicos"""
    model = XGBoostModel()
    predictions = model.predict_multiple_days(days, loja_id)
    
    prediction_values = [pred['prediction'] for pred in predictions]
    weather_data = [pred.get('weather_data', {}) for pred in predictions]
    
    # Estatísticas
    stats = {}
    if prediction_values:
        stats = {
            "mean": round(np.mean(prediction_values), 2),
            "median": round(np.median(prediction_values), 2),
            "std": round(np.std(prediction_values), 2),
            "min": round(min(prediction_values), 2),
            "max": round(max(prediction_values), 2),
            "total": round(sum(prediction_values), 2)
        }
    
    return {
        "days": days,
        "loja_id": loja_id,
        "predictions": prediction_values,
        "weather_forecast": weather_data,
        "detailed_predictions": predictions,
        "statistics": stats,
        "model_status": "loaded" if not model.is_mock else "mock"
    }