import pandas as pd
import sqlite3
import os

# Caminho do CSV consolidado
import glob

# Busca o arquivo forecast_*.csv mais recente
def get_latest_forecast_csv():
    files = glob.glob(os.path.join(os.path.dirname(__file__), 'forecast_*.csv'))
    if not files:
        raise FileNotFoundError("Nenhum arquivo forecast_*.csv encontrado.")
    return max(files, key=os.path.getctime)

csv_path = get_latest_forecast_csv()
# Caminho do banco SQLite
db_path = os.path.join(os.path.dirname(__file__), 'nomads_forecast.db')

def create_table(conn):
    conn.execute('''
    CREATE TABLE IF NOT EXISTS forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        hour INTEGER,
        loja_id INTEGER,
        temp_max REAL,
        temp_min REAL,
        temp_media REAL,
        umid_mediana REAL,
        umid_max REAL,
        umid_min REAL,
        precipitacao_total REAL,
        Chuva_aberta REAL,
        rad_total REAL,
        rad_mediana REAL,
        rad_max REAL,
        rad_min REAL,
        vento_vel_media REAL,
        vento_raj_max REAL,
        dia_semana_num INTEGER,
        day_of_year INTEGER,
        mes INTEGER,
        trimestre INTEGER,
        feriado INTEGER,
        forecast_base TEXT
    );
    ''')
    conn.commit()

def import_csv_to_sqlite(csv_path, db_path):
    # Remove todos os arquivos forecast_*.csv antigos antes de importar
    cache_dir = os.path.dirname(csv_path)
    for f in glob.glob(os.path.join(cache_dir, 'forecast_*.csv')):
        if f != csv_path:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Erro ao remover {f}: {e}")
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    create_table(conn)
    df.to_sql('forecast', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Dados importados para {db_path}")

if __name__ == "__main__":
    import_csv_to_sqlite(csv_path, db_path)
