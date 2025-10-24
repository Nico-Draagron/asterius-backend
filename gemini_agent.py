"""
Asterius Gemini Agent - Agente Inteligente com Google Gemini
Integra Gemini AI, memória e modelo de ML para responder perguntas
"""

import os
import pandas as pd
from typing import Dict, Optional, Any
import google.generativeai as genai
from memory import ConversationMemory
from models import SalesPredictor
from utils import IntentDetector, ResponseFormatter, DateFormatter
from schemas import ChatResponse, VisualizationData
"""from database import AsteriusDB"""


class AsteriusGeminiAgent:
    """
    Agente inteligente do Asterius usando Google Gemini.
    Combina Gemini AI, memória de conversas e modelo de ML.
    """

    def __init__(
        self,
        api_key: str,
    model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        memory: Optional[ConversationMemory] = None,
        predictor: Optional[SalesPredictor] = None
    ):
        """
        Inicializa o agente.
        
        Args:
            api_key: Chave da API Google Gemini
            model_name: Nome do modelo (gemini-1.5-flash, gemini-1.5-pro, etc.)
            temperature: Criatividade das respostas (0-2)
            max_tokens: Máximo de tokens na resposta
            memory: Instância do gerenciador de memória
            predictor: Instância do preditor de vendas
        """
        # Configura a API do Gemini
        genai.configure(api_key=api_key)
        
        # Configura o modelo
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": max_tokens
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Inicializa componentes
        self.memory = memory or ConversationMemory()
        self.predictor = predictor or SalesPredictor()
        # self.db = AsteriusDB()  # Banco desativado temporariamente
        
        # System prompt do agente (contexto local reforçado)
        self.system_prompt = """
Você é o Asterius, um assistente inteligente especializado em análise de vendas, previsões e recomendações para uma loja de açaí e sorvete localizada em Santa Maria - RS, Brasil. Utilize sempre que possível o contexto local, sazonalidade, clima da região e características do negócio.

Suas capacidades:
- Analisar dados de vendas e gerar insights precisos para a loja de açaí e sorvete
- Fazer previsões de vendas futuras usando modelos XGBoost
- Responder perguntas sobre o negócio com dados reais do banco SQLite
- Criar visualizações de dados interativas (gráficos de linha, barra, etc.)
- Integrar dados meteorológicos (chuva, temperatura, umidade) nas análises
- Sugerir ações prescritivas para aumentar as vendas, reduzir perdas ou aproveitar oportunidades

Diretrizes:
- Seja direto e objetivo nas respostas
- Use emojis apropriados para deixar a conversa mais amigável
- Quando falar de valores, use o formato brasileiro (R$ 24.500,00)
- Mantenha um tom profissional mas acessível
- Se não tiver certeza, seja honesto sobre as limitações
- Baseie suas respostas em dados reais sempre que possível
- Considere sempre o contexto de Santa Maria - RS e o segmento de açaí/sorvete

Quando o usuário pedir previsões:
- Use o modelo XGBoost treinado com dados reais
- Inclua informações meteorológicas quando relevante
- Gere números realistas e bem fundamentados
- Explique brevemente a base da previsão
- Destaque pontos importantes (picos, quedas, tendências)
- Mencione correlações com fatores climáticos quando aplicável

Exemplos de perguntas típicas do negócio:
- "Se não chover no fim de semana, quanto posso vender de açaí?"
- "Qual foi o melhor dia de vendas no último verão?"
- "Como a chuva afeta as vendas de sorvete em Santa Maria?"
- "Me mostre um gráfico das vendas nos últimos 30 dias."
- "O que posso fazer para vender mais em dias frios?"
- "Qual a previsão de vendas para a próxima semana?"

Contexto do modelo:
- Modelo: Google Gemini AI integrado com XGBoost
- Dados: Vendas históricas + dados meteorológicos NOMADS/GFS
- Atualização: Dados meteorológicos em tempo real
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
            
            # 2. Verifica se é uma pergunta hipotética
            if self._is_hypothetical_question(message):
                intent = "hypothetical_scenario"
                print(f"🔮 Cenário hipotético detectado")
            
            # 3. Recupera contexto da memória
            context = self.memory.get_context(session_id)
            
            # 4. Adiciona mensagem do usuário à memória
            self.memory.add_message(session_id, "user", message)
            
            # 5. Se for cenário hipotético, chama função específica
            if intent == "hypothetical_scenario":
                return self._handle_hypothetical_scenario(message, context, session_id)
            
            # 6. Se for previsão, chama o modelo ML
            elif intent == "prediction":
                return self._handle_prediction(message, context, session_id)
            
            # 7. Se for consulta de dados, busca no banco
            elif intent == "data_query":
                return self._handle_data_query(message, context, session_id)
            
            # 8. Caso contrário, usa apenas o Gemini
            else:
                return self._handle_general_query(message, context, session_id)
        
        except Exception as e:
            print(f"❌ Erro no agente Gemini: {e}")
            return ChatResponse(
                answer=f"Desculpe, ocorreu um erro ao processar sua mensagem: {str(e)}",
                visualization=None
            )
    
    def _handle_prediction(self, message: str, context: list, session_id: str) -> ChatResponse:
        """
        Trata requisições de previsão de vendas com dados meteorológicos reais.
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
            # Usa o modelo XGBoost com dados reais
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
            
            # Enriquece a resposta com o Gemini incluindo dados meteorológicos
            enriched_answer = self._enrich_prediction_with_gemini(
                predictions, 
                days, 
                message, 
                context,
                weather_data=weather_data,
                model_status="real" if not model.is_mock else "mock"
            )
            
            # Gera visualização baseada na pergunta
            visualization = self._generate_chart_for_question(message, detailed_predictions, weather_data)
            
            # Salva resposta na memória
            self.memory.add_message(session_id, "assistant", enriched_answer)
            
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
    
    def _is_hypothetical_question(self, message: str) -> bool:
        """Detecta se a pergunta é sobre cenários hipotéticos"""
        message_lower = message.lower()
        
        # Palavras-chave para cenários hipotéticos
        hypothetical_keywords = [
            "se não chover", "sem chuva", "se fizesse sol", "se não chovesse",
            "e se", "caso não", "supondo que", "imagine que", "cenário",
            "hipotético", "se o tempo", "se a temperatura", "se tivesse",
            "quanto ganharia", "quanto venderia", "impacto", "diferença"
        ]
        
        return any(keyword in message_lower for keyword in hypothetical_keywords)
    
    def _handle_hypothetical_scenario(self, message: str, context: list, session_id: str) -> ChatResponse:
        """
        Processa perguntas sobre cenários hipotéticos
        """
        try:
            # Extrai modificações da mensagem
            modifications = self._extract_modifications_from_message(message)
            
            # Extrai loja da mensagem (padrão: loja 1)
            loja_id = 1
            if "loja 2" in message.lower() or "segunda loja" in message.lower():
                loja_id = 2
            
            print(f"🔮 Analisando cenário hipotético: {modifications}")
            
            # Usa o modelo XGBoost para fazer a comparação
            from models import XGBoostModel
            model = XGBoostModel()
            
            # Obtém a análise comparativa
            scenario_analysis = model.predict_hypothetical_scenario(
                target_date=None,  # Hoje
                loja_id=loja_id,
                modifications=modifications
            )
            
            if "error" in scenario_analysis:
                return ChatResponse(
                    answer=f"Desculpe, não consegui analisar esse cenário: {scenario_analysis['error']}",
                    visualization=None
                )
            
            # Gera resposta enriquecida com Gemini
            enriched_answer = self._enrich_scenario_with_gemini(
                scenario_analysis, 
                message, 
                context
            )
            
            # Gera visualização de comparação
            visualization = self._generate_scenario_visualization(scenario_analysis)
            
            # Salva resposta na memória
            self.memory.add_message(session_id, "assistant", enriched_answer)
            
            return ChatResponse(
                answer=enriched_answer,
                visualization=visualization
            )
            
        except Exception as e:
            print(f"❌ Erro no cenário hipotético: {e}")
            return ChatResponse(
                answer=f"Desculpe, não consegui analisar esse cenário hipotético: {str(e)}",
                visualization=None
            )
    
    def _extract_modifications_from_message(self, message: str) -> Dict[str, Any]:
        """Extrai modificações das variáveis a partir da mensagem"""
        message_lower = message.lower()
        modifications = {}
        
        # Detecta modificações relacionadas à chuva
        if any(phrase in message_lower for phrase in ["não chover", "sem chuva", "se fizesse sol", "não chovesse"]):
            modifications['Chuva_aberta'] = 0.0
        elif "chuva leve" in message_lower or "pouca chuva" in message_lower:
            modifications['Chuva_aberta'] = 2.0
        elif "chuva forte" in message_lower or "muita chuva" in message_lower:
            modifications['Chuva_aberta'] = 25.0
        
        # Detecta modificações de temperatura
        if "mais quente" in message_lower or "temperatura alta" in message_lower:
            modifications['temp_max'] = 35.0
        elif "mais frio" in message_lower or "temperatura baixa" in message_lower:
            modifications['temp_max'] = 15.0
        elif "calor" in message_lower:
            modifications['temp_max'] = 32.0
        elif "frio" in message_lower:
            modifications['temp_max'] = 18.0
        
        # Detecta modificações de umidade
        if "menos umidade" in message_lower or "seco" in message_lower:
            modifications['umid_mediana'] = 40.0
        elif "mais umidade" in message_lower or "úmido" in message_lower:
            modifications['umid_mediana'] = 85.0
        
        # Se não detectou nenhuma modificação específica, assume "sem chuva" como padrão
        if not modifications and self._is_hypothetical_question(message):
            modifications['Chuva_aberta'] = 0.0
        
        return modifications
    
    def _enrich_scenario_with_gemini(self, scenario_analysis: Dict, message: str, context: list) -> str:
        """Usa Gemini para gerar resposta sobre o cenário hipotético"""
        
        original = scenario_analysis["original_scenario"]
        hypothetical = scenario_analysis["hypothetical_scenario"]
        comparison = scenario_analysis["comparison"]
        
        # Contexto da conversa
        context_text = ""
        if context:
            recent_context = context[-4:]
            context_text = "Contexto da conversa:\n"
            for msg in recent_context:
                role = "Usuário" if msg["role"] == "user" else "Asterius"
                context_text += f"{role}: {msg['content'][:100]}...\n"
        
        prompt = f"""
{self.system_prompt}

{context_text}

ANÁLISE DE CENÁRIO HIPOTÉTICO:

Cenário Atual (Real):
- Previsão de vendas: R$ {original['prediction']:,.2f}
- Condições meteorológicas: {original['weather_data']}

Cenário Hipotético:
- Previsão de vendas: R$ {hypothetical['prediction']:,.2f}
- Modificações aplicadas: {scenario_analysis['modifications_applied']}
- Condições alteradas: {hypothetical['weather_data']}

Comparação:
- Diferença: R$ {comparison['difference']:+,.2f}
- Variação percentual: {comparison['percentage_change']:+.1f}%
- Impacto: {comparison['impact']}
- Análise: {scenario_analysis['scenario_analysis']}

PERGUNTA DO USUÁRIO: "{message}"

INSTRUÇÕES PARA RESPOSTA:
1. Responda diretamente à pergunta hipotética
2. Compare os dois cenários de forma clara
3. Destaque o impacto financeiro da mudança
4. Explique brevemente por que há essa diferença
5. Use emojis apropriados e formatação brasileira
6. Seja concisa mas informativa (máximo 6 linhas)
7. Adicione uma observação prática ou insight relevante

Resposta:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer cenário com Gemini: {e}")
            # Fallback: resposta formatada sem IA
            difference = comparison['difference']
            percentage = comparison['percentage_change']
            impact_emoji = "📈" if difference > 0 else "📉" if difference < 0 else "➡️"
            
            return f"""
{impact_emoji} **Cenário Hipotético Analisado**

🏪 **Cenário Atual**: R$ {original['prediction']:,.2f}
🔮 **Cenário Hipotético**: R$ {hypothetical['prediction']:,.2f}
💰 **Diferença**: R$ {difference:+,.2f} ({percentage:+.1f}%)

{scenario_analysis['scenario_analysis']}
            """.strip()
    
    def _generate_scenario_visualization(self, scenario_analysis: Dict) -> VisualizationData:
        """Gera visualização de comparação entre cenários"""
        try:
            original = scenario_analysis["original_scenario"]
            hypothetical = scenario_analysis["hypothetical_scenario"]
            
            return VisualizationData(
                type="bar",
                labels=["Cenário Atual", "Cenário Hipotético"],
                values=[original['prediction'], hypothetical['prediction']],
                extra={
                    "title": "Comparação de Cenários",
                    "colors": ["#3b82f6", "#10b981"],
                    "currency": True,
                    "comparison": {
                        "difference": scenario_analysis["comparison"]["difference"],
                        "percentage": scenario_analysis["comparison"]["percentage_change"]
                    }
                }
            )
        
        except Exception as e:
            print(f"⚠️ Erro ao gerar visualização de cenário: {e}")
            return None
    
    def _enrich_prediction_with_gemini(
        self, 
        predictions: list, 
        days: int, 
        message: str,
        context: list,
        weather_data: list = None,
        model_status: str = "mock"
    ) -> str:
        """
        Usa o Gemini para gerar uma resposta mais natural sobre a previsão incluindo dados meteorológicos.
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
- Dados obtidos via NOMADS/GFS (modelo meteorológico global)
"""
        
        # Contexto da conversa para o Gemini
        context_text = ""
        if context:
            recent_context = context[-4:]  # Últimas 2 interações
            context_text = "Contexto da conversa:\n"
            for msg in recent_context:
                role = "Usuário" if msg["role"] == "user" else "Asterius"
                context_text += f"{role}: {msg['content'][:100]}...\n"
        
        # Monta prompt para o Gemini
        prompt = f"""
{self.system_prompt}

{context_text}

DADOS DA PREVISÃO:
Previsão para os próximos {days} dias:
- Valores previstos: {[f"R$ {p:,.2f}" for p in predictions]}
- Média do período: R$ {avg:,.2f}
- Pico máximo: R$ {max_val:,.2f}
- Valor mínimo: R$ {min_val:,.2f}
- Modelo usado: {"XGBoost Real com dados meteorológicos" if model_status == "real" else "Simulado"}

{weather_info}

PERGUNTA DO USUÁRIO: "{message}"

INSTRUÇÕES PARA RESPOSTA:
1. Apresente os números principais de forma clara
2. Destaque insights importantes (tendências, picos, variações)
3. Se houver dados meteorológicos, mencione correlações relevantes
4. Seja concisa (máximo 6 linhas)
5. Use emojis apropriados
6. Formate valores no padrão brasileiro
7. Termine com uma observação útil ou recomendação

Resposta:
"""
        
        try:
            # Chama a API Gemini
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer resposta com Gemini: {e}")
            # Fallback: resposta formatada sem IA
            return f"📊 Previsão para {days} dias: Média de R$ {avg:,.2f}, variando entre R$ {min_val:,.2f} e R$ {max_val:,.2f}. Total previsto: R$ {sum(predictions):,.2f}."
    
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
            
            # Gráfico de linha para série temporal (padrão)
            return VisualizationData(
                type="line",
                labels=dates,
                values=sales_values,
                extra={
                    "title": "Previsão de Vendas - Próximos Dias",
                    "colors": ["#3b82f6"],
                    "currency": True
                }
            )
        
        return None
    
    def _handle_data_query(self, message: str, context: list, session_id: str) -> ChatResponse:
        """
        Trata consultas de dados históricos usando Gemini.
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
            
            # Formata resposta com Gemini + visualização
            return self._format_data_response(data, query_info, message, context, session_id)
            
        except Exception as e:
            print(f"❌ Erro na consulta de dados: {e}")
            return ChatResponse(
                answer="Desculpe, houve um erro ao buscar os dados. Tente novamente.",
                visualization=None
            )
    
    def _analyze_data_query(self, message: str) -> Dict:
        """Analisa a consulta para determinar tipo e parâmetros."""
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
            days = IntentDetector.extract_days_from_message(message)
            query_info["days"] = days or 30
        else:
            query_info["type"] = "summary"
        
        return query_info
    
    def _execute_data_query(self, query_info: Dict):
        """Executa a consulta no banco de dados."""
        query_type = query_info["type"]
        loja_id = query_info["loja_id"]
        
        if query_type == "best_days":
            return self.db.get_best_selling_days(limit=query_info["limit"], loja_id=loja_id)
        elif query_type == "weather":
            return self.db.get_weather_correlation(loja_id=loja_id)
        elif query_type == "recent":
            return self.db.get_sales_by_period(days=query_info["days"], loja_id=loja_id)
        elif query_type == "summary":
            return self.db.get_sales_by_period(days=30, loja_id=loja_id)
        else:
            return self.db.get_sales_by_period(days=30, loja_id=loja_id)
    
    def _format_data_response(self, data, query_info: Dict, message: str, context: list, session_id: str) -> ChatResponse:
        """Formata resposta com dados usando Gemini."""
        # Prepara dados para o Gemini
        data_summary = self._summarize_data_for_gemini(data, query_info)
        
        # Gera resposta enriquecida com Gemini
        answer = self._enrich_data_with_gemini(data_summary, message, context)
        
        # Cria visualização se apropriado
        visualization = self._create_data_visualization(data, query_info)
        
        # Salva resposta na memória
        self.memory.add_message(session_id, "assistant", answer)
        
        return ChatResponse(
            answer=answer,
            visualization=visualization
        )
    
    def _summarize_data_for_gemini(self, data, query_info: Dict) -> str:
        """Cria resumo dos dados para o Gemini"""
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
    
    def _enrich_data_with_gemini(self, data_summary: str, message: str, context: list) -> str:
        """Usa Gemini para gerar resposta natural sobre os dados."""
        
        # Contexto da conversa
        context_text = ""
        if context:
            recent_context = context[-4:]
            context_text = "Contexto da conversa:\n"
            for msg in recent_context:
                role = "Usuário" if msg["role"] == "user" else "Asterius"
                context_text += f"{role}: {msg['content'][:100]}...\n"
        
        prompt = f"""
{self.system_prompt}

{context_text}

DADOS ENCONTRADOS:
{data_summary}

PERGUNTA DO USUÁRIO: "{message}"

INSTRUÇÕES:
1. Responda diretamente à pergunta usando os dados
2. Destaque os insights mais importantes
3. Use formatação brasileira para valores (R$ X.XXX,XX)
4. Seja concisa mas informativa (máximo 5 linhas)
5. Use emojis apropriados
6. Se possível, adicione uma recomendação ou observação útil

Resposta:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer resposta de dados com Gemini: {e}")
            return f"Encontrei {data_summary.split('registros')[0]}registros nos dados. {data_summary}"
    
    def _create_data_visualization(self, data, query_info: Dict) -> Optional[VisualizationData]:
        """Cria visualização apropriada para os dados."""
        if data.empty or len(data) < 2:
            return None
        
        try:
            query_type = query_info["type"]
            
            if query_type == "best_days" and 'valores' in data.columns:
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
            
            elif query_type in ["recent", "summary"] and 'data' in data.columns:
                data_sorted = data.sort_values('data').tail(30)
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
        """Trata perguntas gerais usando apenas o Gemini."""
        
        # Contexto da conversa
        context_text = ""
        if context:
            recent_context = context[-6:]  # Últimas 3 interações
            context_text = "Contexto da conversa:\n"
            for msg in recent_context:
                role = "Usuário" if msg["role"] == "user" else "Asterius"
                context_text += f"{role}: {msg['content'][:150]}...\n"
        
        prompt = f"""
{self.system_prompt}

{context_text}

PERGUNTA DO USUÁRIO: "{message}"

INSTRUÇÕES:
- Responda como Asterius, assistente especializado em vendas
- Seja útil, direto e amigável
- Use emojis apropriados
- Se a pergunta for sobre capacidades, mencione que usa Gemini AI + XGBoost + dados meteorológicos
- Máximo 6 linhas

Resposta:
"""
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Salva resposta na memória
            self.memory.add_message(session_id, "assistant", answer)
            
            return ChatResponse(
                answer=answer,
                visualization=None
            )
        
        except Exception as e:
            print(f"❌ Erro na API Gemini: {e}")
            return ChatResponse(
                answer="Desculpe, estou com dificuldades para processar sua mensagem no momento. Tente novamente.",
                visualization=None
            )
    
    def get_capabilities(self) -> Dict:
        """Retorna informações sobre as capacidades do agente"""
        return {
            "model": self.model_name,
            "ai_provider": "Google Gemini",
            "ml_model_loaded": self.predictor.is_loaded,
            "memory_sessions": self.memory.get_session_count(),
            "max_prediction_days": 30,
            "features": [
                "Previsões com XGBoost",
                "Dados meteorológicos NOMADS/GFS",
                "Análise de correlações climáticas",
                "Visualizações interativas",
                "Memória de conversas"
            ]
        }
    
    def __repr__(self):
        return f"<AsteriusGeminiAgent model={self.model_name} sessions={self.memory.get_session_count()}>"