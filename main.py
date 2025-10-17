"""
Asterius Backend - FastAPI Application
Backend inteligente para análise de vendas com IA
"""

import os
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Importa módulos locais
from agent import AsteriusAgent
from gemini_agent import AsteriusGeminiAgent
from memory import ConversationMemory
from models import SalesPredictor
from schemas import (
    ChatRequest, ChatResponse,
    PredictRequest, PredictResponse,
    HealthResponse, ErrorResponse
)
from utils import validate_session_id, DateFormatter
from NOMADS.captar_nomads import NOMADSDownloader
import asyncio

# Carrega variáveis de ambiente
load_dotenv()

# Configura logger raiz para usar saída console em UTF-8 (evita UnicodeEncodeError no Windows)
import logging
import sys
import io

root_logger = logging.getLogger()
# Remove handlers existentes para evitar duplicação ao recarregar em dev
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

# Envolve stdout.buffer em TextIOWrapper com encoding utf-8 e errors='replace'
console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
console_handler = logging.StreamHandler(console_stream)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

# ============================================
# CONFIGURAÇÕES GLOBAIS
# ============================================

# Variáveis de ambiente
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "modelo/modelo_final_xgb.pkl")
HISTORY_PATH = os.getenv("HISTORY_PATH", "backend/data/history.json")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# Instâncias globais (serão inicializadas no lifespan)
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from agent import AsteriusAgent
    from gemini_agent import AsteriusGeminiAgent
    from memory import ConversationMemory
    from models import SalesPredictor

agent: Union["AsteriusAgent", "AsteriusGeminiAgent", None] = None  # Pode ser AsteriusAgent ou AsteriusGeminiAgent
memory: "ConversationMemory | None" = None
predictor: "SalesPredictor | None" = None


# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Inicializa recursos na startup e limpa no shutdown.
    """
    # Startup
    print("\n" + "="*50)
    print("🚀 Inicializando Asterius Backend...")
    print("="*50)
    
    global agent, memory, predictor
    
    # Inicializa componentes
    try:
        memory = ConversationMemory(history_path=HISTORY_PATH)
        predictor = SalesPredictor(model_path=ML_MODEL_PATH)

        # Escolhe o provedor de IA baseado na configuração
        if AI_PROVIDER.lower() == "gemini":
            if not GEMINI_API_KEY:
                print("⚠️ AVISO: GEMINI_API_KEY não configurada!")
                print("📝 Configure no arquivo .env para usar o Gemini")
                raise ValueError("GEMINI_API_KEY é obrigatória para usar o Gemini")

            agent = AsteriusGeminiAgent(
                api_key=GEMINI_API_KEY,
                model_name=GEMINI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                memory=memory,
                predictor=predictor
            )
            print(f"🤖 Provedor de IA: Google Gemini ({GEMINI_MODEL})")

        else:  # OpenAI (padrão)
            if not OPENAI_API_KEY:
                print("⚠️ AVISO: OPENAI_API_KEY não configurada!")
                print("📝 Configure no arquivo .env para usar o GPT")
                # Em desenvolvimento, permite continuar com chave dummy
                if ENVIRONMENT != "development":
                    raise ValueError("OPENAI_API_KEY é obrigatória em produção")

            agent = AsteriusAgent(
                api_key=OPENAI_API_KEY or "dummy-key",
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                memory=memory,
                predictor=predictor
            )
            print(f"🤖 Provedor de IA: OpenAI ({MODEL_NAME})")

        print(f"🔧 Caminho do modelo: {ML_MODEL_PATH}")
        print(f"✅ Agente inicializado com sucesso")
        print(f"✅ Memória: {memory.get_session_count()} sessões")
        print(f"✅ Modelo ML: {'XGBoost Real' if predictor.is_loaded else 'Mock'}")
        print("="*50)

        # FORÇA ATUALIZAÇÃO NOMADS AO INICIAR BACKEND
        print("🌐 Baixando/preparando forecast NOMADS mais recente...")
        downloader = NOMADSDownloader()
        await downloader.download_forecast_range(days_ahead=5)
        downloader.cleanup_old_files(days_to_keep=30)
        print("✅ Forecast NOMADS atualizado!")

        print("🎯 Backend pronto para receber requisições!")
        print("="*50 + "\n")
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        raise
    
    yield
    
    # Shutdown
    print("\n" + "="*50)
    print("👋 Encerrando Asterius Backend...")
    print("="*50 + "\n")


# ============================================
# FASTAPI APP
# ============================================


from hourly_weather_router import router as hourly_weather_router

app = FastAPI(
    title="Asterius Backend API",
    description="Backend inteligente para análise de vendas com IA",
    version="1.0.0",
    lifespan=lifespan
)

# Registrar router de dados horários NOMADS
app.include_router(hourly_weather_router)

# ============================================
# CORS MIDDLEWARE
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# EXCEPTION HANDLERS
# ============================================

from fastapi import Request

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler customizado para HTTPException"""
    err = ErrorResponse(
        error=exc.detail,
        detail=None
    )
    # Serializa datetime para string
    content = err.model_dump()
    if hasattr(err, 'timestamp') and isinstance(content.get('timestamp'), (datetime,)):
        content['timestamp'] = content['timestamp'].isoformat()
    return JSONResponse(
        status_code=exc.status_code,
        content=content
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções gerais"""
    print(f"❌ Erro não tratado: {exc}")
    err = ErrorResponse(
        error="Erro interno do servidor",
        detail=str(exc) if ENVIRONMENT == "development" else None
    )
    content = err.model_dump()
    if hasattr(err, 'timestamp') and isinstance(content.get('timestamp'), (datetime,)):
        content['timestamp'] = content['timestamp'].isoformat()
    return JSONResponse(
        status_code=500,
        content=content
    )


# ============================================
# ENDPOINTS
from api_analise_utils import prever_e_classificar, exportar_para_api
from fastapi import Request

@app.post("/api/analise", tags=["Análise Sazonal"])
async def api_analise(request: Request):
    try:
        dados = await request.json()
        resultado = prever_e_classificar(dados)
        api_json = exportar_para_api(resultado)
        return JSONResponse(content=api_json)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz - informações básicas da API"""
    return HealthResponse(
        status="ok",
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Endpoint principal do chatbot.
    
    Processa mensagens do usuário e retorna respostas inteligentes.
    Pode incluir visualizações de dados quando apropriado.
    """
    # Valida session_id
    if not validate_session_id(request.session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id inválido. Use apenas letras, números, traços e underscores."
        )
    
    try:
        response = agent.respond(
            message=request.message,
            session_id=request.session_id
        )
        print(f"✅ Resposta gerada | Visualização: {response.visualization is not None}")
        return response
    except Exception as e:
        print(f"❌ Erro no chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar mensagem: {str(e)}"
        )


@app.get("/weather/{days}/{loja_id}", tags=["Weather"])
async def get_weather_forecast(days: int = 5, loja_id: int = 1):
    """
    Endpoint para obter previsão meteorológica do NOMADS.
    
    Args:
        days: Número de dias de previsão (1-7)
        loja_id: ID da loja (1 ou 2)
        
    Returns:
        Dados meteorológicos formatados
    """
    try:
        from models import XGBoostModel

        if days < 1 or days > 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Número de dias deve estar entre 1 e 7"
            )

        if loja_id not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID da loja deve ser 1 ou 2"
            )

        model = XGBoostModel()
        weather_data = model.get_weather_forecast(days, loja_id)

        print(f"🌤️ Dados meteorológicos obtidos para {days} dias, loja {loja_id}")

        return {
            "success": True,
            "days": days,
            "loja_id": loja_id,
            "data_source": "NOMADS/GFS",
            "weather_data": weather_data
        }
    except Exception as e:
        print(f"❌ Erro ao obter dados meteorológicos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter dados meteorológicos: {str(e)}"
        )


@app.get("/predictions-with-weather/{days}/{loja_id}", tags=["Prediction"])
async def get_predictions_with_weather(days: int = 5, loja_id: int = 1):
    """
    Endpoint para obter previsões de vendas com dados meteorológicos.
    
    Args:
        days: Número de dias de previsão (1-7)
        loja_id: ID da loja (1 ou 2)
        
    Returns:
        Previsões de vendas com dados meteorológicos correlacionados
    """
    try:
        from models import XGBoostModel
        import importlib.util
        import asyncio

        if days < 1 or days > 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Número de dias deve estar entre 1 e 7"
            )

        if loja_id not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID da loja deve ser 1 ou 2"
            )

        from datetime import datetime, timedelta, date as date_cls
        model = XGBoostModel()
        predictions = model.predict_multiple_days(days, loja_id)

        # Gera ciclo de datas: ontem, hoje, +5
        today = datetime.now().date()
        cycle_dates = [(today - timedelta(days=1)) + timedelta(days=i) for i in range(7)]
        cycle_strs = [d.strftime('%Y-%m-%d') for d in cycle_dates]

        # Indexar previsões por data
        pred_map = {p.get('weather_data', {}).get('date', ''): p for p in predictions}

        # Se faltar dados NOMADS para algum dos dias, executa captar_nomads
        missing_dates = [date_str for date_str in cycle_strs if date_str not in pred_map or not pred_map[date_str].get('weather_data')]
        if missing_dates:
            print(f"⚠️ Faltam dados NOMADS para: {missing_dates}. Executando captar_nomads...")
            # Caminho absoluto para captar_nomads.py
            captar_path = os.path.join(os.path.dirname(__file__), "NOMADS", "captar_nomads.py")
            import importlib.util
            import asyncio
            spec = importlib.util.spec_from_file_location("captar_nomads", captar_path)
            captar_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(captar_module)
            # Executa o main async do script para baixar os dados faltantes
            await captar_module.main()
            # Recarrega o modelo e as previsões após baixar os dados
            model = XGBoostModel()
            predictions = model.predict_multiple_days(days, loja_id)
            pred_map = {p.get('weather_data', {}).get('date', ''): p for p in predictions}

        def to_str(val):
            if isinstance(val, (datetime, date_cls)):
                return val.isoformat()
            return val

        formatted_data: dict[str, Any] = {
            "success": True,
            "days": 7,
            "loja_id": loja_id,
            "model_status": "real" if not model.is_mock else "mock",
            "predictions": [],
            "weather_data": [],
            "charts": {
                "sales": {"labels": [], "data": []},
                "temperature": {"labels": [], "data": []},
                "precipitation": {"labels": [], "data": []}
            }
        }

        for date_str in cycle_strs:
            pred = pred_map.get(date_str)
            if pred:
                weather = pred.get('weather_data', {})
                value = pred['prediction']
                confidence = pred.get('confidence', 0.75)
            else:
                weather = {"date": date_str}
                value = None
                confidence = None
            formatted_data["predictions"].append({
                "date": to_str(date_str),
                "value": value,
                "confidence": confidence
            })
            formatted_data["weather_data"].append({
                "date": to_str(date_str),
                "temperature": weather.get('temp_media', None),
                "temp_max": weather.get('temp_max', None),
                "humidity": weather.get('umid_mediana', None),
                "precipitation": weather.get('precipitacao_total', None),
                "radiation": weather.get('rad_max', None)
            })
            # Dados para gráficos
            sales_chart = formatted_data["charts"]["sales"]
            temp_chart = formatted_data["charts"]["temperature"]
            precip_chart = formatted_data["charts"]["precipitation"]
            sales_chart["labels"].append(to_str(date_str))
            sales_chart["data"].append(value)
            temp_chart["labels"].append(to_str(date_str))
            temp_chart["data"].append(weather.get('temp_media', None))
            precip_chart["labels"].append(to_str(date_str))
            precip_chart["data"].append(weather.get('precipitacao_total', None))

        print(f"📊🌤️ Previsões com dados meteorológicos geradas para 7 dias, loja {loja_id}")
        return formatted_data
    except Exception as e:
        print(f"❌ Erro ao gerar previsões com dados meteorológicos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar previsões: {str(e)}"
        )


@app.get("/sessions", tags=["Memory"])
async def get_sessions():
    """
    Lista todas as sessões ativas.
    Útil para debug e monitoramento.
    """
    sessions = memory.get_all_sessions()
    return {
        "total": len(sessions),
        "sessions": sessions
    }

    
    history = memory.get_context(session_id)
    
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sessão {session_id} não encontrada"
        )

    return {
        "session_id": session_id,
        "message_count": len(history),
        "messages": history
    }


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """
    Retorna estatísticas do sistema.
    """
    return {
        "agent": agent.get_capabilities(),
        "memory": {
            "total_sessions": memory.get_session_count(),
            "sessions": memory.get_all_sessions()
        },
        "predictor": {
            "model_loaded": predictor.is_loaded,
            "model_path": predictor.model_path
        }
    }


# ============================================
# MAIN - Para rodar com python main.py
# ============================================

@app.get("/hourly-precipitation/{date}", tags=["Weather"])
async def get_hourly_precipitation(date: str):
    """
    Retorna dados de precipitação por hora para um dia específico
    
    Args:
        date: Data no formato YYYY-MM-DD
        
    Returns:
        Dados de precipitação por hora (24 horas)
    """
    try:
        import random
        import math
        from datetime import datetime
        # Por enquanto, vamos gerar dados mockados
        # No futuro, isso pode ser expandido para usar dados NOMADS horários reais
        hourly_data = []
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            # Simular padrão de precipitação realista
            precipitation = 0.0
            if 13 <= hour <= 18:  # Tarde (mais provável)
                precipitation = max(0, (hour - 10) * 0.8 + (hour - 15) ** 2 * 0.2)
            elif 20 <= hour <= 23:  # Noite
                precipitation = max(0, (23 - hour) * 0.5)
            elif 2 <= hour <= 6:  # Madrugada (menos provável)
                precipitation = max(0, (6 - hour) * 0.3)
            # Adicionar variação aleatória
            precipitation += random.uniform(-0.5, 1.5)
            precipitation = max(0, precipitation)  # Não pode ser negativo
            # Temperatura simulada (varia ao longo do dia)
            base_temp = 22
            temp_variation = 8 * math.sin((hour - 6) * math.pi / 12)
            temperature = base_temp + temp_variation + random.uniform(-2, 2)
            hourly_data.append({
                "hour": hour_str,
                "precipitation": round(precipitation, 1),
                "temperature": round(temperature, 1)
            })
        # Use string for timestamp to avoid JSON serialization error
        # Find peak_hour safely (handle empty list)
        peak_hour = None
        if hourly_data:
            peak = max(hourly_data, key=lambda x: float(x["precipitation"]))
            peak_hour = peak["hour"]
        return {
            "success": True,
            "date": date,
            "data": hourly_data,
            "total_precipitation": sum(item["precipitation"] for item in hourly_data),
            "peak_hour": peak_hour,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Erro ao obter dados horários de precipitação: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter dados horários: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"\n🚀 Iniciando servidor em http://{host}:{port}")
    print(f"📚 Documentação em http://{host}:{port}/docs\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True if ENVIRONMENT == "development" else False
    )