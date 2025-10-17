"""
Memory Management do Asterius
Gerencia o histórico de conversas por sessão de usuário
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from schemas import Message


class ConversationMemory:
    """
    Gerencia a memória de conversas do agente.
    Mantém as últimas 5 mensagens por sessão e persiste em disco.
    """
    
    def __init__(self, history_path: str = "data/history.json", max_messages: int = 5):
        """
        Inicializa o gerenciador de memória.
        
        Args:
            history_path: Caminho para o arquivo de histórico JSON
            max_messages: Número máximo de mensagens a manter por sessão (padrão: 5)
        """
        self.history_path = history_path
        self.max_messages = max_messages
        self.conversations: Dict[str, List[Dict]] = {}
        
        # Cria o diretório se não existir
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Carrega histórico existente
        self._load_history()
    
    def _load_history(self):
        """Carrega o histórico do arquivo JSON"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    self.conversations = json.load(f)
                print(f"✅ Histórico carregado: {len(self.conversations)} sessões")
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao carregar histórico. Criando novo arquivo.")
                self.conversations = {}
        else:
            print(f"📝 Arquivo de histórico não encontrado. Será criado em: {self.history_path}")
            self.conversations = {}
    
    def _save_history(self):
        """Salva o histórico no arquivo JSON"""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"❌ Erro ao salvar histórico: {e}")
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        Adiciona uma mensagem ao histórico da sessão.
        
        Args:
            session_id: ID da sessão do usuário
            role: 'user' ou 'assistant'
            content: Conteúdo da mensagem
        """
        # Cria a sessão se não existir
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Adiciona a mensagem
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations[session_id].append(message)
        
        # Mantém apenas as últimas N mensagens
        if len(self.conversations[session_id]) > self.max_messages * 2:  # *2 porque user + assistant
            self.conversations[session_id] = self.conversations[session_id][-(self.max_messages * 2):]
        
        # Salva no disco
        self._save_history()
    
    def get_context(self, session_id: str) -> List[Dict]:
        """
        Retorna o contexto (últimas mensagens) de uma sessão.
        
        Args:
            session_id: ID da sessão do usuário
            
        Returns:
            Lista de mensagens no formato {"role": "...", "content": "..."}
        """
        return self.conversations.get(session_id, [])
    
    def get_recent_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retorna as N mensagens mais recentes de uma sessão.
        
        Args:
            session_id: ID da sessão
            limit: Número máximo de mensagens (None = todas)
            
        Returns:
            Lista de mensagens
        """
        messages = self.conversations.get(session_id, [])
        if limit:
            return messages[-limit:]
        return messages
    
    def clear_session(self, session_id: str):
        """
        Limpa o histórico de uma sessão específica.
        
        Args:
            session_id: ID da sessão a limpar
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            self._save_history()
            print(f"🗑️ Histórico da sessão {session_id} removido")
    
    def get_all_sessions(self) -> List[str]:
        """Retorna lista de todas as sessões ativas"""
        return list(self.conversations.keys())
    
    def get_session_count(self) -> int:
        """Retorna o número total de sessões"""
        return len(self.conversations)
    
    def get_message_count(self, session_id: str) -> int:
        """Retorna o número de mensagens em uma sessão"""
        return len(self.conversations.get(session_id, []))
    
    def format_context_for_llm(self, session_id: str) -> str:
        """
        Formata o contexto da sessão para enviar ao LLM.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            String formatada com o histórico da conversa
        """
        messages = self.get_context(session_id)
        
        if not messages:
            return "Nenhuma conversa anterior."
        
        formatted = "Histórico da conversa:\n\n"
        for msg in messages:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            formatted += f"{role}: {msg['content']}\n"
        
        return formatted
    
    def __repr__(self):
        return f"<ConversationMemory sessions={self.get_session_count()} path={self.history_path}>"