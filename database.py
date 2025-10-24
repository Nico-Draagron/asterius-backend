"""
Database Manager do Asterius
Gerencia importação de CSVs para SQLite e consultas inteligentes
"""

import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional

load_dotenv()

class AsteriusDB:
    def get_hourly_weather(self, date: str, loja_id: int) -> pd.DataFrame:
        """
        Retorna dados horários para uma data e loja específica.
        Args:
            date: Data no formato YYYY-MM-DD
            loja_id: ID da loja
        Returns:
            DataFrame com dados horários
        """
        sql = f"""
        SELECT * FROM weather_hourly
        WHERE DATE(date) = '{date}' AND loja_id = {loja_id}
        ORDER BY hour
        """
        return self.query(sql)

    def import_hourly_weather(self, df: pd.DataFrame):
        """
        Importa DataFrame de dados horários para a tabela 'weather_hourly'.
        Remove dados antigos e insere os novos.
        """
        # Cria ou substitui tabela na primeira importação
        df.to_sql('weather_hourly', self.engine, if_exists='replace', index=False)
        print(f"✅ Dados horários importados para 'weather_hourly': {len(df)} registros")
    """
    Gerenciador de banco de dados do Asterius.
    Importa CSVs e permite consultas inteligentes.
    """
    
    def __init__(self, db_url: str = None):
        """
        Inicializa o gerenciador de banco.
        
        Args:
            db_url: URL da database (usa DATABASE_URL do .env se não fornecido)
        """
        self.db_url = db_url or os.getenv('DATABASE_URL', 'sqlite:///./asterius.db')
        self.engine = create_engine(self.db_url)
        self.datasets_path = Path('datasets')
        
        print(f"🗄️ Conectado ao banco: {self.db_url}")
    
    def import_csvs(self, force_reimport: bool = False):
        """
        Importa todos os CSVs da pasta datasets para o banco.
        
        Args:
            force_reimport: Se True, reimporta mesmo se já existir
        """
        print("📥 Iniciando importação de CSVs...")
        
        csv_files = {
            'loja1_dados': 'Loja1_processado.csv',
            'loja2_dados': 'Loja2_processado.csv'
        }
        
        for table_name, csv_file in csv_files.items():
            csv_path = self.datasets_path / csv_file
            
            if not csv_path.exists():
                print(f"⚠️ Arquivo não encontrado: {csv_path}")
                continue
            
            # Verifica se tabela já existe
            if not force_reimport and self._table_exists(table_name):
                print(f"ℹ️ Tabela '{table_name}' já existe. Use force_reimport=True para reimportar.")
                continue
            
            print(f"📊 Importando {csv_file} -> {table_name}")
            
            try:
                # Lê CSV
                df = pd.read_csv(csv_path)
                
                # Trata dados
                df = self._clean_dataframe(df, table_name)
                
                # Importa para banco
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
                
                print(f"✅ {table_name}: {len(df)} registros importados")
                
                # Mostra info da tabela
                self._show_table_info(table_name, df)
                
            except Exception as e:
                print(f"❌ Erro ao importar {csv_file}: {e}")
        
        print("🎉 Importação concluída!")
    
    def _clean_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Limpa e prepara o DataFrame para importação.
        
        Args:
            df: DataFrame original
            table_name: Nome da tabela
            
        Returns:
            DataFrame limpo
        """
        # Adiciona ID da loja baseado no nome da tabela
        if 'loja1' in table_name:
            df['loja_id'] = 1
        elif 'loja2' in table_name:
            df['loja_id'] = 2
        
        # Converte data para datetime
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'])
        
        # Remove colunas vazias, duplicadas ou problemáticas
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Remove colunas com nomes problemáticos (como 'trimestre.1')
        df = df.loc[:, ~df.columns.str.contains(r'\.')]
        
        # Define colunas padrão que devem existir
        standard_columns = [
            'data', 'valores', 'pedidos', 'temp_max', 'temp_min', 'temp_media',
            'umid_max', 'umid_min', 'umid_mediana', 'rad_min', 'rad_max', 'rad_mediana',
            'vento_raj_max', 'vento_vel_media', 'precipitacao_total', 'dia_semana_num',
            'dia_semana_binario', 'day_of_year', 'mes', 'trimestre', 'Chuva_aberta',
            'dia_mes', 'feriado', 'vespera_feriado', 'semana_ano', 'loja_id'
        ]
        
        # Mantém apenas colunas que existem e estão na lista padrão
        existing_columns = [col for col in standard_columns if col in df.columns]
        df = df[existing_columns]
        
        # Trata valores nulos em colunas numéricas
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def _table_exists(self, table_name: str) -> bool:
        """Verifica se tabela existe no banco"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"))
                return result.fetchone() is not None
        except:
            return False
    
    def _show_table_info(self, table_name: str, df: pd.DataFrame):
        """Mostra informações da tabela importada"""
        print(f"📋 Info da tabela '{table_name}':")
        print(f"   📊 Registros: {len(df)}")
        print(f"   📅 Período: {df['data'].min()} a {df['data'].max()}")
        if 'valores' in df.columns:
            total_vendas = df['valores'].sum()
            print(f"   💰 Total vendas: R$ {total_vendas:,.2f}")
        print()
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Executa query SQL e retorna DataFrame.
        
        Args:
            sql: Query SQL
            
        Returns:
            DataFrame com resultados
        """
        try:
            return pd.read_sql(sql, self.engine)
        except Exception as e:
            print(f"❌ Erro na query: {e}")
            return pd.DataFrame()
    
    def get_sales_by_period(self, days: int = 30, loja_id: int = None) -> pd.DataFrame:
        """
        Retorna vendas dos últimos N dias.
        
        Args:
            days: Número de dias
            loja_id: ID da loja (None = todas)
            
        Returns:
            DataFrame com vendas
        """
        where_clause = ""
        if loja_id:
            where_clause = f"AND loja_id = {loja_id}"
        
        sql = f"""
        SELECT 
            data,
            loja_id,
            valores,
            pedidos,
            temp_media,
            umid_mediana,
            precipitacao_total
        FROM (
            SELECT * FROM loja1_dados
            UNION ALL
            SELECT * FROM loja2_dados
        ) 
        WHERE data >= date('now', '-{days} days')
        {where_clause}
        ORDER BY data DESC
        """
        
        return self.query(sql)
    
    def get_weather_correlation(self, loja_id: int = None) -> pd.DataFrame:
        """
        Analisa correlação entre clima e vendas.
        
        Args:
            loja_id: ID da loja (None = todas)
            
        Returns:
            DataFrame com correlações
        """
        where_clause = ""
        if loja_id:
            where_clause = f"WHERE loja_id = {loja_id}"
        
        sql = f"""
        SELECT 
            ROUND(temp_media, 0) as temperatura,
            ROUND(AVG(valores), 2) as vendas_media,
            COUNT(*) as dias,
            ROUND(AVG(precipitacao_total), 2) as chuva_media
        FROM (
            SELECT * FROM loja1_dados
            UNION ALL
            SELECT * FROM loja2_dados
        )
        {where_clause}
        GROUP BY ROUND(temp_media, 0)
        HAVING COUNT(*) >= 5
        ORDER BY temperatura
        """
        
        return self.query(sql)
    
    def get_best_selling_days(self, limit: int = 10, loja_id: int = None) -> pd.DataFrame:
        """
        Retorna os melhores dias de venda.
        
        Args:
            limit: Número de registros
            loja_id: ID da loja
            
        Returns:
            DataFrame com melhores dias
        """
        where_clause = ""
        if loja_id:
            where_clause = f"WHERE loja_id = {loja_id}"
        
        sql = f"""
        SELECT 
            data,
            loja_id,
            valores,
            pedidos,
            temp_media,
            dia_semana_num,
            precipitacao_total
        FROM (
            SELECT * FROM loja1_dados
            UNION ALL
            SELECT * FROM loja2_dados
        )
        {where_clause}
        ORDER BY valores DESC
        LIMIT {limit}
        """
        
        return self.query(sql)
    
    def search_by_date_range(self, start_date: str, end_date: str, loja_id: int = None) -> pd.DataFrame:
        """
        Busca vendas por período específico.
        
        Args:
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            loja_id: ID da loja
            
        Returns:
            DataFrame com resultados
        """
        where_clause = ""
        if loja_id:
            where_clause = f"AND loja_id = {loja_id}"
        
        sql = f"""
        SELECT * FROM (
            SELECT * FROM loja1_dados
            UNION ALL
            SELECT * FROM loja2_dados
        )
        WHERE data BETWEEN '{start_date}' AND '{end_date}'
        {where_clause}
        ORDER BY data
        """
        
        return self.query(sql)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas resumidas do banco.
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {}
        
        # Estatísticas gerais
        sql = """
        SELECT 
            COUNT(*) as total_registros,
            COUNT(DISTINCT loja_id) as total_lojas,
            MIN(data) as data_inicio,
            MAX(data) as data_fim,
            SUM(valores) as vendas_total,
            AVG(valores) as vendas_media
        FROM (
            SELECT * FROM loja1_dados
            UNION ALL
            SELECT * FROM loja2_dados
        )
        WHERE valores > 0
        """
        
        result = self.query(sql)
        if not result.empty:
            stats.update(result.iloc[0].to_dict())
        
        # Estatísticas por loja
        sql = """
        SELECT 
            loja_id,
            COUNT(*) as registros,
            SUM(valores) as vendas_total,
            AVG(valores) as vendas_media,
            MAX(valores) as melhor_dia
        FROM (
            SELECT * FROM loja1_dados
            UNION ALL
            SELECT * FROM loja2_dados
        )
        WHERE valores > 0
        GROUP BY loja_id
        """
        
        stats['por_loja'] = self.query(sql).to_dict('records')
        
        return stats
    
    def list_tables(self) -> List[str]:
        """Lista todas as tabelas do banco"""
        sql = "SELECT name FROM sqlite_master WHERE type='table';"
        result = self.query(sql)
        return result['name'].tolist() if not result.empty else []
    
    def __repr__(self):
        tables = self.list_tables()
        return f"<AsteriusDB tables={tables} url={self.db_url}>"


# ============================================
# FUNÇÕES DE CONVENIÊNCIA
# ============================================

def setup_database(force_reimport: bool = False):
    """
    Configura o banco de dados importando os CSVs.
    
    Args:
        force_reimport: Força reimportação mesmo se já existir
    """
    db = AsteriusDB()
    db.import_csvs(force_reimport=force_reimport)
    return db

def get_sales_summary():
    """Retorna resumo das vendas"""
    db = AsteriusDB()
    return db.get_summary_stats()

def search_sales(query_type: str, **kwargs) -> pd.DataFrame:
    """
    Busca vendas baseado no tipo de consulta.
    
    Args:
        query_type: Tipo da consulta ('period', 'weather', 'best_days', 'date_range')
        **kwargs: Argumentos específicos para cada tipo
        
    Returns:
        DataFrame com resultados
    """
    db = AsteriusDB()
    
    if query_type == 'period':
        return db.get_sales_by_period(kwargs.get('days', 30), kwargs.get('loja_id'))
    elif query_type == 'weather':
        return db.get_weather_correlation(kwargs.get('loja_id'))
    elif query_type == 'best_days':
        return db.get_best_selling_days(kwargs.get('limit', 10), kwargs.get('loja_id'))
    elif query_type == 'date_range':
        return db.search_by_date_range(kwargs.get('start_date'), kwargs.get('end_date'), kwargs.get('loja_id'))
    else:
        print(f"❌ Tipo de consulta inválido: {query_type}")
        return pd.DataFrame()