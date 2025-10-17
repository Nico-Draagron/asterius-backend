"""
Asterius Agent - Agente Inteligente com OpenAI
Integra LLM, memória e modelo de ML para responder perguntas
"""

import os
import pandas as pd
from typing import Dict, Optional
from openai import OpenAI
from memory import ConversationMemory
from models import SalesPredictor
from utils import IntentDetector, ResponseFormatter, DateFormatter
from schemas import ChatResponse, VisualizationData
"""from database import AsteriusDB"""


class AsteriusAgent:
    """
    Agente inteligente do Asterius.
    Combina OpenAI GPT, memória de conversas e modelo de ML.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        memory: Optional[ConversationMemory] = None,
        predictor: Optional[SalesPredictor] = None
    ):
        """
        Inicializa o agente.
        
        Args:
            api_key: Chave da API OpenAI
            model_name: Nome do modelo (gpt-4, gpt-4o, etc.)
            temperature: Criatividade das respostas (0-1)
            max_tokens: Máximo de tokens na resposta
            memory: Instância do gerenciador de memória
            predictor: Instância do preditor de vendas
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Inicializa componentes
        self.memory = memory or ConversationMemory()
        self.predictor = predictor or SalesPredictor()
    # self.db = AsteriusDB()  # Banco desativado temporariamente
        
        # System prompt do agente
        self.system_prompt = """
Você é o Asterius, um assistente inteligente especializado em análise de vendas e previsões.

Suas capacidades:
- Analisar dados de vendas e gerar insights
- Fazer previsões de vendas futuras
- Responder perguntas sobre o negócio
- Criar visualizações de dados

Diretrizes:
- Seja direto e objetivo nas respostas
- Use emojis apropriados para deixar a conversa mais amigável
- Quando falar de valores, use o formato brasileiro (R$ 24.500,00)
- Mantenha um tom profissional mas acessível
- Se não tiver certeza, seja honesto sobre as limitações

Quando o usuário pedir previsões:
- Gere números realistas e bem fundamentados
- Explique brevemente a base da previsão
- Destaque pontos importantes (picos, quedas, tendências)
"""
    
    def respond(self, message: str, session_id: str) -> ChatResponse:
        """
        Processa uma mensagem e retorna resposta.
        
        Args:
            message: Mensagem do usuário
            session_id: ID da sessão
            
        Returns:
            ChatResponse com resposta e possível visualização
        """
        try:
            # 1. Detecta a intenção
            intent = IntentDetector.detect_intent(message)
            print(f"🎯 Intenção detectada: {intent}")
            
            # 2. Recupera contexto da memória
            context = self.memory.get_context(session_id)
            
            # 3. Adiciona mensagem do usuário à memória
            self.memory.add_message(session_id, "user", message)
            
            # 4. Se for previsão, chama o modelo ML
            if intent == "prediction":
                return self._handle_prediction(message, context)
            
            # 5. Se for consulta de dados, busca no banco
            elif intent == "data_query":
                return self._handle_data_query(message, context, session_id)
            
            # 6. Caso contrário, usa apenas o LLM
            else:
                return self._handle_general_query(message, context, session_id)
        
        except Exception as e:
            print(f"❌ Erro no agente: {e}")
            return ChatResponse(
                answer=f"Desculpe, ocorreu um erro ao processar sua mensagem: {str(e)}",
                visualization=None
            )
    
    def _handle_prediction(self, message: str, context: list) -> ChatResponse:
        """
        Trata requisições de previsão de vendas com dados meteorológicos reais.
        
        Args:
            message: Mensagem do usuário
            context: Contexto da conversa
            
        Returns:
            ChatResponse com previsão e visualização
        """
        # Extrai número de dias da mensagem
        days = IntentDetector.extract_days_from_message(message)
        if days is None:
            days = 7  # Padrão: 7 dias
        
        # Extrai loja da mensagem (padrão: loja 1)
        loja_id = 1
        if "loja 2" in message.lower() or "segunda loja" in message.lower():
            loja_id = 2
        
        print(f"📊 Gerando previsão para {days} dias, loja {loja_id}")
        
        try:
            # Usa o novo modelo XGBoost com dados reais
            from models import XGBoostModel
            model = XGBoostModel()
            
            # Obtém predições detalhadas com dados meteorológicos
            detailed_predictions = model.predict_multiple_days(days, loja_id)
            
            # Extrai valores de predição para compatibilidade
            predictions = [pred['prediction'] for pred in detailed_predictions]
            
            # Extrai dados meteorológicos reais
            weather_data = []
            for pred in detailed_predictions:
                weather = pred.get('weather_data', {})
                weather_data.append({
                    "date": weather.get('date', ''), 
                    "temperature": weather.get('temp_media', 22.0),
                    "temp_max": weather.get('temp_max', 25.0),
                    "humidity": weather.get('umid_mediana', 65.0),
                    "precipitation": weather.get('precipitacao_total', 0.0),
                    "rain": weather.get('Chuva_aberta', 0.0)
                })
            
            # Formata resposta com visualização incluindo dados meteorológicos
            response_data = ResponseFormatter.format_prediction_response(predictions, days)
            
            # Adiciona dados meteorológicos à visualização
            if response_data.get("visualization"):
                response_data["visualization"]["weather_data"] = weather_data
                response_data["visualization"]["charts"]["weather"] = {
                    "temperature": [w["temperature"] for w in weather_data],
                    "precipitation": [w["precipitation"] for w in weather_data],
                    "labels": [w["date"] or f"Dia {i+1}" for i, w in enumerate(weather_data)]
                }
            
            # Enriquece a resposta com o LLM incluindo dados meteorológicos
            enriched_answer = self._enrich_prediction_with_llm(
                predictions, 
                days, 
                message, 
                context,
                weather_data=weather_data,
                model_status="real" if not model.is_mock else "mock"
            )
            
            # Gera visualização baseada na pergunta
            visualization = self._generate_chart_for_question(message, detailed_predictions, weather_data)
            
            return ChatResponse(
                answer=enriched_answer,
                visualization=visualization
            )
        
        except Exception as e:
            print(f"❌ Erro na previsão: {e}")
            return ChatResponse(
                answer=f"Desculpe, não consegui gerar a previsão: {str(e)}",
                visualization=None
            )
    
    def _generate_chart_for_question(self, message: str, predictions: list, weather_data: list = None) -> VisualizationData:
        """
        Gera dados de visualização baseados na pergunta do usuário
        """
        message_lower = message.lower()
        
        # Prepara dados básicos
        dates = [pred.get('date', f"Dia {i+1}") for i, pred in enumerate(predictions)]
        sales_values = [pred.get('prediction', {}).get('vendas_previstas', 0) for pred in predictions]
        
        # Identifica tipo de gráfico baseado na pergunta
        if any(word in message_lower for word in ['gráfico', 'grafico', 'chart', 'visualiz', 'mostr']):
            
            # Gráfico de linha para série temporal
            if any(word in message_lower for word in ['série temporal', 'serie temporal', 'linha', 'tendência', 'tendencia']):
                return VisualizationData(
                    type="line",
                    labels=dates,
                    values=sales_values,
                    extra={
                        "title": "Previsão de Vendas - Série Temporal",
                        "colors": ["#3b82f6"]
                    }
                )
            
            # Gráfico de barras para comparação
            elif any(word in message_lower for word in ['barras', 'bar', 'compar']):
                return VisualizationData(
                    type="bar",
                    labels=dates,
                    values=sales_values,
                    extra={
                        "title": "Previsão de Vendas por Dia",
                        "colors": ["#10b981"]
                    }
                )
            
            # Boxplot para distribuição
            elif any(word in message_lower for word in ['boxplot', 'distribuição', 'distribuicao']):
                return VisualizationData(
                    type="boxplot",
                    labels=["Vendas Previstas"],
                    values=sales_values,
                    extra={
                        "title": "Distribuição das Vendas Previstas",
                        "colors": ["#8b5cf6"]
                    }
                )
            
            # Gráfico de área
            elif any(word in message_lower for word in ['área', 'area']):
                return VisualizationData(
                    type="area",
                    labels=dates,
                    values=sales_values,
                    extra={
                        "title": "Tendência de Vendas",
                        "colors": ["#f59e0b"]
                    }
                )
            
            # Scatter plot para correlação
            elif any(word in message_lower for word in ['scatter', 'dispersão', 'correlação']):
                return VisualizationData(
                    type="scatter",
                    labels=dates,
                    values=sales_values,
                    extra={
                        "title": "Vendas vs Fatores Climáticos",
                        "colors": ["#ef4444"]
                    }
                )
            
            # Gráfico com dados meteorológicos
            elif weather_data and any(word in message_lower for word in ['clima', 'tempo', 'meteorológic', 'chuva', 'temperatura']):
                temp_values = [w.get('temperature', 22) for w in weather_data]
                return VisualizationData(
                    type="line",
                    labels=dates,
                    values=sales_values,
                    extra={
                        "title": "Vendas vs Temperatura",
                        "datasets": [
                            {
                                "label": "Temperatura (°C)",
                                "data": temp_values,
                                "color": "#ef4444"
                            }
                        ]
                    }
                )
            
            # Default: gráfico de linha
            else:
                return VisualizationData(
                    type="line",
                    labels=dates,
                    values=sales_values,
                    extra={
                        "title": "Previsão de Vendas",
                        "colors": ["#3b82f6"]
                    }
                )
        
        return None
    
    def _enrich_prediction_with_llm(
        self, 
        predictions: list, 
        days: int, 
        message: str,
        context: list,
        weather_data: list = None,
        model_status: str = "mock"
    ) -> str:
        """
        Usa o LLM para gerar uma resposta mais natural sobre a previsão incluindo dados meteorológicos.
        
        Args:
            predictions: Valores previstos
            days: Número de dias
            message: Mensagem original
            context: Contexto da conversa
            weather_data: Dados meteorológicos
            model_status: Status do modelo (real/mock)
            
        Returns:
            Resposta enriquecida
        """
        # Calcula estatísticas
        avg = sum(predictions) / len(predictions)
        max_val = max(predictions)
        min_val = min(predictions)
        
        # Prepara informações meteorológicas
        weather_info = ""
        if weather_data:
            avg_temp = sum(w.get('temperature', 0) for w in weather_data) / len(weather_data)
            total_rain = sum(w.get('precipitation', 0) for w in weather_data)
            weather_info = f"""
Condições meteorológicas para o período:
- Temperatura média: {avg_temp:.1f}°C
- Precipitação total: {total_rain:.1f}mm
- Dados obtidos via NOMADS/GFS
"""
        
        # Monta prompt para o LLM
        prompt = f"""
Com base na previsão de vendas e dados meteorológicos abaixo, gere uma resposta natural e útil para o usuário.

Previsão para os próximos {days} dias:
- Valores: {predictions}
- Média: R$ {avg:,.2f}
- Máximo: R$ {max_val:,.2f}
- Mínimo: R$ {min_val:,.2f}
- Modelo usado: {"XGBoost Real" if model_status == "real" else "Simulado"}

{weather_info}

Pergunta do usuário: "{message}"

Gere uma resposta que:
1. Apresente os números principais
2. Destaque insights importantes (tendências, picos, correlação com clima)
3. Mencione se há correlação entre clima e vendas
4. Seja concisa (máximo 6 linhas)
5. Use emojis apropriados

Resposta:
"""
        
        # Monta mensagens para a API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Chama a API OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer resposta: {e}")
            # Fallback: resposta formatada sem LLM
            return ResponseFormatter.format_prediction_response(predictions, days)["answer"]
    
    def _handle_data_query(self, message: str, context: list, session_id: str) -> ChatResponse:
        """
        Trata consultas de dados históricos.
        
        Args:
            message: Mensagem do usuário
            context: Contexto da conversa
            session_id: ID da sessão
            
        Returns:
            ChatResponse com dados e possível visualização
        """
        try:
            # Detecta tipo de consulta específica
            query_info = self._analyze_data_query(message)
            
            # Executa consulta no banco
            data = self._execute_data_query(query_info)
            
            if data is None or data.empty:
                return ChatResponse(
                    answer="Desculpe, não encontrei dados para sua consulta. Tente reformular a pergunta.",
                    visualization=None
                )
            
            # Formata resposta com LLM + visualização
            return self._format_data_response(data, query_info, message, context, session_id)
            
        except Exception as e:
            print(f"❌ Erro na consulta de dados: {e}")
            return ChatResponse(
                answer="Desculpe, houve um erro ao buscar os dados. Tente novamente.",
                visualization=None
            )
    
    def _analyze_data_query(self, message: str) -> Dict:
        """
        Analisa a consulta para determinar tipo e parâmetros.
        
        Args:
            message: Mensagem do usuário
            
        Returns:
            Dicionário com informações da consulta
        """
        message_lower = message.lower()
        
        query_info = {
            "type": "general",
            "loja_id": None,
            "days": None,
            "limit": 10
        }
        
        # Detecta loja específica
        if "loja 1" in message_lower or "primeira loja" in message_lower:
            query_info["loja_id"] = 1
        elif "loja 2" in message_lower or "segunda loja" in message_lower:
            query_info["loja_id"] = 2
        
        # Detecta tipo de consulta
        if any(word in message_lower for word in ["melhores", "maiores", "top", "recordes"]):
            query_info["type"] = "best_days"
        elif any(word in message_lower for word in ["piores", "menores", "baixas"]):
            query_info["type"] = "worst_days"
        elif any(word in message_lower for word in ["clima", "temperatura", "chuva", "correlação"]):
            query_info["type"] = "weather"
        elif any(word in message_lower for word in ["últimos", "recentes", "período"]):
            query_info["type"] = "recent"
            # Extrai número de dias
            days = IntentDetector.extract_days_from_message(message)
            query_info["days"] = days or 30
        else:
            query_info["type"] = "summary"
        
        return query_info
    
    def _execute_data_query(self, query_info: Dict):
        """
        Executa a consulta no banco de dados.
        
        Args:
            query_info: Informações da consulta
            
        Returns:
            DataFrame com resultados
        """
        query_type = query_info["type"]
        loja_id = query_info["loja_id"]
        
        if query_type == "best_days":
            return self.db.get_best_selling_days(limit=query_info["limit"], loja_id=loja_id)
        elif query_type == "weather":
            return self.db.get_weather_correlation(loja_id=loja_id)
        elif query_type == "recent":
            return self.db.get_sales_by_period(days=query_info["days"], loja_id=loja_id)
        elif query_type == "summary":
            # Retorna resumo das últimas semanas
            return self.db.get_sales_by_period(days=30, loja_id=loja_id)
        else:
            return self.db.get_sales_by_period(days=30, loja_id=loja_id)
    
    def _format_data_response(self, data, query_info: Dict, message: str, context: list, session_id: str) -> ChatResponse:
        """
        Formata resposta com dados + visualização usando LLM.
        
        Args:
            data: DataFrame com dados
            query_info: Informações da consulta
            message: Mensagem original
            context: Contexto da conversa
            session_id: ID da sessão
            
        Returns:
            ChatResponse formatada
        """
        # Prepara dados para o LLM
        data_summary = self._summarize_data_for_llm(data, query_info)
        
        # Gera resposta enriquecida com LLM
        answer = self._enrich_data_with_llm(data_summary, message, context)
        
        # Cria visualização se apropriado
        visualization = self._create_data_visualization(data, query_info)
        
        # Salva resposta na memória
        self.memory.add_message(session_id, "assistant", answer)
        
        return ChatResponse(
            answer=answer,
            visualization=visualization
        )
    
    def _summarize_data_for_llm(self, data, query_info: Dict) -> str:
        """Cria resumo dos dados para o LLM"""
        if data.empty:
            return "Nenhum dado encontrado."
        
        summary = f"Dados encontrados: {len(data)} registros\n"
        
        if 'valores' in data.columns:
            valores = data['valores'][data['valores'] > 0]
            if not valores.empty:
                summary += f"Vendas - Total: R$ {valores.sum():,.2f}, Média: R$ {valores.mean():,.2f}\n"
                summary += f"Melhor dia: R$ {valores.max():,.2f}, Pior dia: R$ {valores.min():,.2f}\n"
        
        if 'data' in data.columns:
            summary += f"Período: {data['data'].min()} a {data['data'].max()}\n"
        
        # Adiciona top 3 resultados para contexto
        if query_info["type"] == "best_days" and len(data) > 0:
            summary += "Top 3 dias:\n"
            for i, row in data.head(3).iterrows():
                summary += f"  {row.get('data', 'N/A')}: R$ {row.get('valores', 0):,.2f}\n"
        
        return summary
    
    def _enrich_data_with_llm(self, data_summary: str, message: str, context: list) -> str:
        """
        Usa LLM para gerar resposta natural sobre os dados.
        
        Args:
            data_summary: Resumo dos dados
            message: Mensagem original
            context: Contexto da conversa
            
        Returns:
            Resposta enriquecida
        """
        prompt = f"""
Com base nos dados abaixo, responda à pergunta do usuário de forma natural e útil.

Dados encontrados:
{data_summary}

Pergunta: "{message}"

Gere uma resposta que:
1. Responda diretamente à pergunta
2. Destaque os insights mais importantes
3. Use formatação brasileira para valores (R$ X.XXX,XX)
4. Seja concisa mas informativa
5. Use emojis apropriados

Resposta:
"""
        
        # Monta mensagens para a API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer resposta de dados: {e}")
            # Fallback: resposta simples
            return f"Encontrei {data_summary.split('registros')[0]}registros nos dados. {data_summary}"
    
    def _create_data_visualization(self, data, query_info: Dict) -> Optional[VisualizationData]:
        """
        Cria visualização apropriada para os dados.
        
        Args:
            data: DataFrame com dados
            query_info: Informações da consulta
            
        Returns:
            VisualizationData ou None
        """
        if data.empty or len(data) < 2:
            return None
        
        try:
            query_type = query_info["type"]
            
            if query_type == "best_days" and 'valores' in data.columns:
                # Gráfico de barras dos melhores dias
                data_viz = data.head(10)
                labels = [str(d)[:10] if pd.notna(d) else f"Dia {i+1}" for i, d in enumerate(data_viz.get('data', range(len(data_viz))))]
                values = data_viz['valores'].fillna(0).tolist()
                
                return VisualizationData(
                    type="bar",
                    labels=labels,
                    values=values,
                    extra={
                        "title": "Melhores Dias de Vendas",
                        "currency": True
                    }
                )
            
            elif query_type == "weather" and 'temperatura' in data.columns:
                # Gráfico de dispersão temperatura x vendas
                labels = [f"{temp}°C" for temp in data['temperatura']]
                values = data['vendas_media'].fillna(0).tolist()
                
                return VisualizationData(
                    type="line",
                    labels=labels,
                    values=values,
                    extra={
                        "title": "Vendas por Temperatura",
                        "currency": True
                    }
                )
            
            elif query_type in ["recent", "summary"] and 'data' in data.columns:
                # Gráfico de linha temporal
                data_sorted = data.sort_values('data').tail(30)  # Últimos 30 registros
                labels = [str(d)[:10] if pd.notna(d) else f"Dia {i+1}" for i, d in enumerate(data_sorted.get('data', range(len(data_sorted))))]
                values = data_sorted['valores'].fillna(0).tolist()
                
                return VisualizationData(
                    type="line", 
                    labels=labels,
                    values=values,
                    extra={
                        "title": "Vendas no Período",
                        "currency": True
                    }
                )
            
        except Exception as e:
            print(f"⚠️ Erro ao criar visualização: {e}")
        
        return None
    
    def _handle_general_query(self, message: str, context: list, session_id: str) -> ChatResponse:
        """
        Trata perguntas gerais usando apenas o LLM.
        
        Args:
            message: Mensagem do usuário
            context: Contexto da conversa
            session_id: ID da sessão
            
        Returns:
            ChatResponse com resposta textual
        """
        # Monta mensagens com contexto
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Adiciona contexto (últimas 5 mensagens)
        for msg in context[-10:]:  # Últimas 5 interações (user + assistant)
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Adiciona mensagem atual
        messages.append({"role": "user", "content": message})
        
        try:
            # Chama a API OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Salva resposta na memória
            self.memory.add_message(session_id, "assistant", answer)
            
            return ChatResponse(
                answer=answer,
                visualization=None
            )
        
        except Exception as e:
            print(f"❌ Erro na API OpenAI: {e}")
            return ChatResponse(
                answer="Desculpe, estou com dificuldades para processar sua mensagem no momento. Tente novamente.",
                visualization=None
            )
    
    def get_capabilities(self) -> Dict:
        """Retorna informações sobre as capacidades do agente"""
        return {
            "model": self.model_name,
            "ml_model_loaded": self.predictor.is_loaded,
            "memory_sessions": self.memory.get_session_count(),
            "max_prediction_days": 30
        }
    
    def __repr__(self):
        return f"<AsteriusAgent model={self.model_name} sessions={self.memory.get_session_count()}>"