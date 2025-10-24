"""
Utility Functions do Asterius
Funções auxiliares para detecção de intenção e formatação
"""

import re
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class IntentDetector:
    """
    Detector de intenções nas mensagens dos usuários.
    Identifica se o usuário quer previsões, análises, etc.
    """
    
    # Palavras-chave para detecção de intenções
    INTENT_KEYWORDS = {
        "prediction": [
            "previsão", "prever", "projeção", "estimativa", "futuro",
            "amanhã", "próximo", "próximos dias", "próxima semana",
            "vai vender", "vou vender", "expectativa", "tendência"
        ],
        "data_query": [
            "vendas", "dados", "histórico", "buscar", "mostrar", "listar",
            "qual foi", "quanto", "quando", "melhores dias", "piores dias",
            "correlação", "clima", "temperatura", "chuva", "últimos",
            "período", "janeiro", "fevereiro", "março", "abril", "maio", "junho"
        ],
        "analysis": [
            "análise", "analisar", "avaliar", "comparar", "desempenho",
            "performance", "relatório", "dashboard", "métricas"
        ],
        "explanation": [
            "explique", "explicar", "como funciona", "o que é", "me diga",
            "qual é", "quem é", "quando", "por que", "porque"
        ],
        "greeting": [
            "olá", "oi", "bom dia", "boa tarde", "boa noite",
            "hey", "hello", "hi"
        ]
    }
    
    @staticmethod
    def detect_intent(message: str) -> str:
        """
        Detecta a intenção principal da mensagem.
        
        Args:
            message: Mensagem do usuário
            
        Returns:
            String com a intenção: 'prediction', 'analysis', 'explanation', 'greeting', 'general'
        """
        message_lower = message.lower()
        
        # Verifica cada tipo de intenção
        for intent, keywords in IntentDetector.INTENT_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return "general"
    
    @staticmethod
    def extract_days_from_message(message: str) -> Optional[int]:
        """
        Extrai o número de dias mencionado na mensagem.
        
        Args:
            message: Mensagem do usuário
            
        Returns:
            Número de dias ou None se não encontrado
        """
        # Padrões para detectar números
        patterns = [
            r'(\d+)\s*dias?',
            r'próximos?\s*(\d+)',
            r'para\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                days = int(match.group(1))
                # Limita entre 1 e 30 dias
                return min(max(days, 1), 30)
        
        # Detecta palavras específicas
        if "amanhã" in message.lower():
            return 1
        elif "semana" in message.lower():
            return 7
        elif "mês" in message.lower() or "mes" in message.lower():
            return 30
        
        return None
    
    @staticmethod
    def should_generate_visualization(intent: str) -> bool:
        """
        Determina se deve gerar visualização baseado na intenção.
        
        Args:
            intent: Intenção detectada
            
        Returns:
            True se deve gerar visualização
        """
        return intent in ["prediction", "analysis"]


class DateFormatter:
    """Formatador de datas para labels de gráficos"""
    
    @staticmethod
    def generate_date_labels(days: int, start_date: Optional[datetime] = None) -> list:
        """
        Gera labels de datas para gráficos.
        
        Args:
            days: Número de dias
            start_date: Data inicial (padrão: hoje)
            
        Returns:
            Lista de strings formatadas (ex: ["06/10", "07/10"])
        """
        if start_date is None:
            start_date = datetime.now()
        
        labels = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            labels.append(date.strftime("%d/%m"))
        
        return labels
    
    @staticmethod
    def format_currency(value: float) -> str:
        """
        Formata valor como moeda brasileira.
        
        Args:
            value: Valor numérico
            
        Returns:
            String formatada (ex: "R$ 24.500,00")
        """
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


class ResponseFormatter:
    """Formatador de respostas para o frontend"""
    
    @staticmethod
    def format_prediction_response(values: list, days: Optional[int] = None) -> Dict:
        """
        Formata resposta de previsão com visualização.
        
        Args:
            values: Lista de valores previstos
            days: Número de dias (opcional, usa len(values) se não fornecido)
            
        Returns:
            Dicionário com answer e visualization
        """
        if days is None:
            days = len(values)
        
        labels = DateFormatter.generate_date_labels(days)
        
        # Calcula estatísticas
        avg_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)
        
        # Monta resposta textual
        answer = f"Aqui está a previsão de vendas para os próximos {days} dias:\n\n"
        answer += f"📊 Média prevista: {DateFormatter.format_currency(avg_value)}\n"
        answer += f"📈 Maior valor: {DateFormatter.format_currency(max_value)}\n"
        answer += f"📉 Menor valor: {DateFormatter.format_currency(min_value)}\n"
        
        # Monta visualização
        visualization = {
            "type": "line",
            "labels": labels,
            "values": values,
            "extra": {
                "title": f"Previsão de Vendas - Próximos {days} dias",
                "currency": True,
                "avg": avg_value,
                "max": max_value,
                "min": min_value
            }
        }
        
        return {
            "answer": answer,
            "visualization": visualization
        }
    
    @staticmethod
    def format_error_response(error: str, detail: Optional[str] = None) -> Dict:
        """
        Formata resposta de erro.
        
        Args:
            error: Mensagem de erro principal
            detail: Detalhes adicionais (opcional)
            
        Returns:
            Dicionário com erro formatado
        """
        response = {
            "answer": f"❌ Desculpe, ocorreu um erro: {error}",
            "visualization": None
        }
        
        if detail:
            response["answer"] += f"\n\nDetalhes: {detail}"
        
        return response


def validate_session_id(session_id: str) -> bool:
    """
    Valida se o session_id está em formato válido.
    
    Args:
        session_id: ID da sessão
        
    Returns:
        True se válido
    """
    if not session_id or len(session_id) < 3:
        return False
    
    # Permite apenas caracteres alfanuméricos, traços e underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', session_id))


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Trunca texto longo para evitar sobrecarga.
    
    Args:
        text: Texto a truncar
        max_length: Tamanho máximo
        
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def extract_numbers_from_text(text: str) -> list:
    """
    Extrai todos os números de um texto.
    
    Args:
        text: Texto contendo números
        
    Returns:
        Lista de floats encontrados
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]