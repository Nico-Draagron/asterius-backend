"""
Schemas do Asterius Backend
Define os modelos Pydantic para validação de requests e responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================
# REQUEST MODELS
# ============================================

class ChatRequest(BaseModel):
    """Request para o endpoint /chat"""
    session_id: str = Field(..., description="ID único da sessão do usuário")
    message: str = Field(..., min_length=1, description="Mensagem do usuário")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123",
                "message": "Qual a previsão de vendas para amanhã?"
            }
        }


class PredictRequest(BaseModel):
    """Request para o endpoint /predict"""
    days: int = Field(default=7, ge=1, le=30, description="Número de dias para prever (1-30)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "days": 7
            }
        }


# ============================================
# RESPONSE MODELS
# ============================================

class VisualizationData(BaseModel):
    """Dados para visualização em gráficos"""
    type: str = Field(..., description="Tipo do gráfico: line, bar, pie, etc.")
    labels: List[str] = Field(..., description="Labels para o eixo X")
    values: List[float] = Field(..., description="Valores para o eixo Y")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Dados extras como título, cores, etc.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "line",
                "labels": ["Seg", "Ter", "Qua"],
                "values": [20000, 23000, 24500],
                "extra": {"title": "Previsão de vendas - próximos 3 dias"}
            }
        }


class ChatResponse(BaseModel):
    """Response do endpoint /chat"""
    answer: str = Field(..., description="Resposta textual do agente")
    visualization: Optional[VisualizationData] = Field(default=None, description="Dados de visualização opcional")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da resposta")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "A previsão de vendas para amanhã é de R$ 24.500.",
                "visualization": {
                    "type": "line",
                    "labels": ["Seg", "Ter", "Qua"],
                    "values": [20000, 23000, 24500],
                    "extra": {"title": "Previsão de vendas - próximos 3 dias"}
                },
                "timestamp": "2025-10-06T10:30:00"
            }
        }
        ser_json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictResponse(BaseModel):
    """Response do endpoint /predict"""
    labels: List[str] = Field(..., description="Datas previstas")
    values: List[float] = Field(..., description="Valores previstos")
    
    class Config:
        json_schema_extra = {
            "example": {
                "labels": ["06/10", "07/10", "08/10"],
                "values": [21000, 23000, 24500]
            }
        }


class HealthResponse(BaseModel):
    """Response do endpoint /health"""
    status: str = Field(..., description="Status do servidor")
    version: str = Field(..., description="Versão da API")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": "2025-10-06T10:30:00"
            }
        }
        ser_json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Response padrão para erros"""
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(default=None, description="Detalhes adicionais do erro")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Modelo de ML não encontrado",
                "detail": "O arquivo sales_model.pkl não existe no diretório ml_models/",
                "timestamp": "2025-10-06T10:30:00"
            }
        }
        ser_json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================
# INTERNAL MODELS
# ============================================

class Message(BaseModel):
    """Modelo interno para mensagens no histórico"""
    role: str = Field(..., description="Papel: 'user' ou 'assistant'")
    content: str = Field(..., description="Conteúdo da mensagem")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Qual a previsão para amanhã?",
                "timestamp": "2025-10-06T10:30:00"
            }
        }
        ser_json_encoders = {
            datetime: lambda v: v.isoformat()
        }