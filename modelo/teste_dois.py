import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import joblib
import os
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================
seed = 42
np.random.seed(seed)

# Determinar diretórios baseado na localização atual
current_dir = os.path.basename(os.getcwd())
if current_dir == "modelo":
    # Se já estamos na pasta modelo, datasets estão em ../datasets e salvamos na pasta atual
    datasets_path = "../datasets"
    model_dir = "."
else:
    # Se estamos na raiz, datasets estão em ./datasets e criamos pasta modelo
    datasets_path = "datasets"
    model_dir = "modelo"
    os.makedirs(model_dir, exist_ok=True)

print("="*70)
print("MODELO EM DOIS ESTÁGIOS: SAZONALIDADE + CLIMA")
print("="*70)
print(f"Executando de: {os.getcwd()}")
print(f"Datasets em: {os.path.abspath(datasets_path)}")
print(f"Modelos serão salvos em: {os.path.abspath(model_dir)}")

# =============================================================================
# CARREGAR E CONCATENAR DATASETS
# =============================================================================
print("\n--- Carregando Datasets ---")

# Verificar se os arquivos existem
loja1_path = f'{datasets_path}/Loja1_processado.csv'
loja2_path = f'{datasets_path}/Loja2_processado.csv'

if not os.path.exists(loja1_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {loja1_path}")
if not os.path.exists(loja2_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {loja2_path}")

loja1 = pd.read_csv(loja1_path)
loja1['loja_id'] = 1
loja2 = pd.read_csv(loja2_path)
loja2['loja_id'] = 2

# Harmonizar colunas
loja1 = loja1.loc[:, ~loja1.columns.duplicated()]
loja2 = loja2.loc[:, ~loja2.columns.duplicated()]

common_cols = [col for col in loja1.columns if col in loja2.columns]
loja1 = loja1[common_cols]
loja2 = loja2[common_cols]

# Concatenar
entrada = pd.concat([loja1, loja2], ignore_index=True)
print(f"Dataset concatenado: {entrada.shape[0]} linhas, {entrada.shape[1]} colunas")

# =============================================================================
# PREPROCESSAMENTO TEMPORAL E CRIAÇÃO DE FEATURES
# =============================================================================
print("\n--- Criando Features Temporais e Sazonais ---")

# Converter coluna 'data' para datetime e ordenar
entrada['data'] = pd.to_datetime(entrada['data'])
entrada = entrada.sort_values('data').reset_index(drop=True)

# Garantir valores numéricos
for col in ['mes', 'day_of_year', 'dia_mes']:
    if col in entrada.columns:
        entrada[col] = pd.to_numeric(entrada[col], errors='coerce').astype('Int64')

# Preencher NaN em 'mes' e 'day_of_year'
entrada['mes'] = entrada['mes'].fillna(1).astype(int)
entrada['day_of_year'] = entrada['day_of_year'].fillna(1).astype(int)

# 1. CRIAR FEATURES CÍCLICAS PARA MÊS
entrada['mes_sin'] = np.sin(2 * np.pi * entrada['mes'] / 12)
entrada['mes_cos'] = np.cos(2 * np.pi * entrada['mes'] / 12)
print("✓ Features cíclicas de mês criadas: mes_sin, mes_cos")

# 2. CRIAR FEATURES CÍCLICAS PARA DIA DO ANO
entrada['day_sin'] = np.sin(2 * np.pi * entrada['day_of_year'] / 365)
entrada['day_cos'] = np.cos(2 * np.pi * entrada['day_of_year'] / 365)
print("✓ Features cíclicas de dia do ano criadas: day_sin, day_cos")

# 3. EXTRAIR SAZONALIDADE VIA SEASONAL_DECOMPOSE
print("✓ Extraindo componente sazonal via seasonal_decompose...")

# Agregar série mensal
monthly_series = entrada.set_index('data')['valores'].resample('M').sum()

if len(monthly_series) < 24:
    warnings.warn("Série mensal tem menos de 24 pontos. Decomposição pode ser instável.")

# Decomposição sazonal
decomp = seasonal_decompose(monthly_series, period=12, model='additive', extrapolate_trend='freq')
seasonal_monthly = decomp.seasonal

# Mapear sazonalidade para cada linha
entrada['month_end'] = entrada['data'].dt.to_period('M').dt.to_timestamp('M')

seasonal_df = seasonal_monthly.reset_index()
seasonal_df.columns = ['data_seasonal', 'sazonalidade']

entrada = entrada.merge(seasonal_df, how='left', left_on='month_end', right_on='data_seasonal')

# Remover colunas temporárias
if 'data_seasonal' in entrada.columns:
    entrada.drop(columns=['data_seasonal'], inplace=True)

# Preencher NaN na sazonalidade usando média mensal
if 'sazonalidade' in entrada.columns and entrada['sazonalidade'].isna().any():
    seasonal_by_month = {}
    for idx, val in seasonal_monthly.items():
        month = idx.month
        if month not in seasonal_by_month:
            seasonal_by_month[month] = []
        seasonal_by_month[month].append(val)
    
    month_means = {month: np.mean(values) for month, values in seasonal_by_month.items()}
    
    entrada['sazonalidade'] = entrada.apply(
        lambda r: month_means.get(r['mes'], 0.0) if pd.isna(r.get('sazonalidade', np.nan)) else r['sazonalidade'], 
        axis=1
    )

if 'sazonalidade' not in entrada.columns:
    entrada['sazonalidade'] = 0.0
else:
    entrada['sazonalidade'] = entrada['sazonalidade'].fillna(0.0)

# Limpar colunas
entrada.drop(columns=['month_end'], inplace=True, errors='ignore')

print(f"✓ Componente sazonal extraído e mapeado")

# Remover NaN da variável alvo
entrada.dropna(subset=['valores'], inplace=True)

# =============================================================================
# MODELO 1: SAZONALIDADE → VENDAS
# =============================================================================
print("\n" + "="*70)
print("MODELO 1: SAZONALIDADE → VENDAS")
print("="*70)

# Selecionar apenas features sazonais
df_sazonal = entrada[[
    'valores', 'sazonalidade', 'mes_sin', 'mes_cos', 'day_sin', 'day_cos', 'loja_id'
]].copy()

df_sazonal['loja_id'] = df_sazonal['loja_id'].astype('category')

X_sazonal = df_sazonal.drop('valores', axis=1)
y = df_sazonal['valores']

print(f"✓ Features sazonais: {X_sazonal.shape[1]} variáveis")
print(f"  - sazonalidade, mes_sin, mes_cos, day_sin, day_cos, loja_id")

# Definir pré-processador para modelo sazonal
numeric_features_sazonal = ['sazonalidade', 'mes_sin', 'mes_cos', 'day_sin', 'day_cos']
categorical_features_sazonal = ['loja_id']

numeric_transformer_sazonal = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer_sazonal = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor_sazonal = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_sazonal, numeric_features_sazonal),
        ('cat', categorical_transformer_sazonal, categorical_features_sazonal)
    ])

# --- FASE 1.1: OTIMIZAÇÃO MODELO SAZONALIDADE ---
print("\n--- Fase 1.1: Otimização de Hiperparâmetros (Modelo Sazonalidade) ---")

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sazonal, y, test_size=0.2, random_state=25)

def objective_sazonal(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_sazonal),
        ('regressor', xgb.XGBRegressor(random_state=seed, **params))
    ])
    score = cross_val_score(model_pipeline, X_train_s, y_train_s, cv=5, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
    return score.mean()

study_sazonal = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
study_sazonal.optimize(objective_sazonal, n_trials=50)

best_params_sazonal = study_sazonal.best_params
print(f"\n✓ Melhores Hiperparâmetros (Modelo Sazonalidade):")
for param, value in best_params_sazonal.items():
    print(f"  - {param}: {value}")

# --- FASE 1.2: BOOTSTRAP MODELO SAZONALIDADE ---
print("\n--- Fase 1.2: Avaliação com Bootstrap (Modelo Sazonalidade) ---")

n_bootstrap_iterations = 100
n_samples = len(df_sazonal)
indices = np.arange(n_samples)

mae_scores_s, rmse_scores_s, r2_scores_s, mape_scores_s = [], [], [], []

for i in tqdm(range(n_bootstrap_iterations), desc="Bootstrap Modelo 1"):
    iteration_seed = seed + i
    
    in_bag_indices = resample(indices, replace=True, n_samples=n_samples, random_state=iteration_seed)
    out_of_bag_indices = np.setdiff1d(indices, np.unique(in_bag_indices))
    
    if len(out_of_bag_indices) == 0:
        continue
    
    X_train_boot = X_sazonal.iloc[in_bag_indices]
    y_train_boot = y.iloc[in_bag_indices]
    X_test_boot = X_sazonal.iloc[out_of_bag_indices]
    y_test_boot = y.iloc[out_of_bag_indices]
    
    model_pipeline_boot = Pipeline(steps=[
        ('preprocessor', preprocessor_sazonal),
        ('regressor', xgb.XGBRegressor(random_state=seed, **best_params_sazonal))
    ])
    
    model_pipeline_boot.fit(X_train_boot, y_train_boot)
    y_pred_boot = model_pipeline_boot.predict(X_test_boot)
    
    mae_scores_s.append(mean_absolute_error(y_test_boot, y_pred_boot))
    rmse_scores_s.append(np.sqrt(mean_squared_error(y_test_boot, y_pred_boot)))
    r2_scores_s.append(r2_score(y_test_boot, y_pred_boot))
    mape_scores_s.append(np.mean(np.abs((y_test_boot - y_pred_boot) / y_test_boot)) * 100)

# Análise resultados Modelo 1
mae_scores_norm_s = np.array(mae_scores_s) / y.max()

def ic_halfwidth(arr):
    ic = np.percentile(arr, [2.5, 97.5])
    return (ic[1] - ic[0]) / 2, ic

med_mae_s = np.median(mae_scores_norm_s)
mae_ic_half_s, mae_ic_s = ic_halfwidth(mae_scores_norm_s)

med_mape_s = np.median(mape_scores_s)
mape_ic_half_s, mape_ic_s = ic_halfwidth(mape_scores_s)

med_r2_s = np.median(r2_scores_s)
r2_ic_half_s, r2_ic_s = ic_halfwidth(r2_scores_s)

print("\n" + "-"*50)
print("RESULTADOS MODELO 1: SAZONALIDADE")
print("-"*50)
print(f"MAE normalizado: {med_mae_s:.4f} ± {mae_ic_half_s:.4f}")
print(f"MAPE: {med_mape_s:.2f}% ± {mape_ic_half_s:.2f}%")
print(f"R²: {med_r2_s:.4f} ± {r2_ic_half_s:.4f}")
print("-"*50)

# Treinar modelo final de sazonalidade com todos os dados
final_model_sazonal = Pipeline(steps=[
    ('preprocessor', preprocessor_sazonal),
    ('regressor', xgb.XGBRegressor(random_state=seed, **best_params_sazonal))
])
final_model_sazonal.fit(X_sazonal, y)

# Gerar predições de sazonalidade para usar no Modelo 2
entrada['sazonalidade_pred'] = final_model_sazonal.predict(X_sazonal)
print("\n✓ Feature 'sazonalidade_pred' criada para o Modelo 2")

# Salvar Modelo 1
modelo_sazonal_path = os.path.join(model_dir, "modelo_1_sazonalidade.pkl")
joblib.dump(final_model_sazonal, modelo_sazonal_path)
print(f"✓ Modelo 1 salvo como '{modelo_sazonal_path}'")

# =============================================================================
# MODELO 2: CLIMA + SAZONALIDADE_PRED → VENDAS
# =============================================================================
print("\n" + "="*70)
print("MODELO 2: CLIMA + SAZONALIDADE_PRED → VENDAS")
print("="*70)

# Selecionar features climáticas + sazonalidade_pred
df_clima = entrada[[
    'valores', 'temp_max', 'temp_media', 'umid_mediana', 'rad_max', 'Chuva_aberta',
    'dia_semana_num', 'feriado', 'trimestre', 'loja_id', 'sazonalidade_pred'
]].copy()

df_clima['dia_semana_num'] = df_clima['dia_semana_num'].astype('category')
df_clima['feriado'] = df_clima['feriado'].astype('category')
df_clima['trimestre'] = df_clima['trimestre'].astype('category')
df_clima['loja_id'] = df_clima['loja_id'].astype('category')

X_clima = df_clima.drop('valores', axis=1)
y_clima = df_clima['valores']

print(f"✓ Features climáticas + sazonalidade_pred: {X_clima.shape[1]} variáveis")
print(f"  - Clima: temp_max, temp_media, umid_mediana, rad_max, Chuva_aberta")
print(f"  - Contexto: dia_semana_num, feriado, trimestre, loja_id")
print(f"  - Sazonalidade: sazonalidade_pred (do Modelo 1)")

# Definir pré-processador para modelo clima
numeric_features_clima = ['temp_max', 'temp_media', 'umid_mediana', 'rad_max', 
                          'Chuva_aberta', 'sazonalidade_pred']
categorical_features_clima = ['dia_semana_num', 'feriado', 'trimestre', 'loja_id']

numeric_transformer_clima = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer_clima = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor_clima = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_clima, numeric_features_clima),
        ('cat', categorical_transformer_clima, categorical_features_clima)
    ])

# --- FASE 2.1: OTIMIZAÇÃO MODELO CLIMA ---
print("\n--- Fase 2.1: Otimização de Hiperparâmetros (Modelo Clima) ---")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clima, y_clima, test_size=0.2, random_state=25)

def objective_clima(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_clima),
        ('regressor', xgb.XGBRegressor(random_state=seed, **params))
    ])
    score = cross_val_score(model_pipeline, X_train_c, y_train_c, cv=5, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
    return score.mean()

study_clima = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
study_clima.optimize(objective_clima, n_trials=50)

best_params_clima = study_clima.best_params
print(f"\n✓ Melhores Hiperparâmetros (Modelo Clima):")
for param, value in best_params_clima.items():
    print(f"  - {param}: {value}")

# --- FASE 2.2: BOOTSTRAP MODELO CLIMA ---
print("\n--- Fase 2.2: Avaliação com Bootstrap (Modelo Clima) ---")

n_samples_clima = len(df_clima)
indices_clima = np.arange(n_samples_clima)

mae_scores_c, rmse_scores_c, r2_scores_c, mape_scores_c = [], [], [], []

for i in tqdm(range(n_bootstrap_iterations), desc="Bootstrap Modelo 2"):
    iteration_seed = seed + i
    
    in_bag_indices = resample(indices_clima, replace=True, n_samples=n_samples_clima, random_state=iteration_seed)
    out_of_bag_indices = np.setdiff1d(indices_clima, np.unique(in_bag_indices))
    
    if len(out_of_bag_indices) == 0:
        continue
    
    X_train_boot = X_clima.iloc[in_bag_indices]
    y_train_boot = y_clima.iloc[in_bag_indices]
    X_test_boot = X_clima.iloc[out_of_bag_indices]
    y_test_boot = y_clima.iloc[out_of_bag_indices]
    
    model_pipeline_boot = Pipeline(steps=[
        ('preprocessor', preprocessor_clima),
        ('regressor', xgb.XGBRegressor(random_state=seed, **best_params_clima))
    ])
    
    model_pipeline_boot.fit(X_train_boot, y_train_boot)
    y_pred_boot = model_pipeline_boot.predict(X_test_boot)
    
    mae_scores_c.append(mean_absolute_error(y_test_boot, y_pred_boot))
    rmse_scores_c.append(np.sqrt(mean_squared_error(y_test_boot, y_pred_boot)))
    r2_scores_c.append(r2_score(y_test_boot, y_pred_boot))
    mape_scores_c.append(np.mean(np.abs((y_test_boot - y_pred_boot) / y_test_boot)) * 100)

# Análise resultados Modelo 2
mae_scores_norm_c = np.array(mae_scores_c) / y_clima.max()

med_mae_c = np.median(mae_scores_norm_c)
mae_ic_half_c, mae_ic_c = ic_halfwidth(mae_scores_norm_c)

med_mape_c = np.median(mape_scores_c)
mape_ic_half_c, mape_ic_c = ic_halfwidth(mape_scores_c)

med_r2_c = np.median(r2_scores_c)
r2_ic_half_c, r2_ic_c = ic_halfwidth(r2_scores_c)

print("\n" + "-"*50)
print("RESULTADOS MODELO 2: CLIMA + SAZONALIDADE")
print("-"*50)
print(f"MAE normalizado: {med_mae_c:.4f} ± {mae_ic_half_c:.4f}")
print(f"MAPE: {med_mape_c:.2f}% ± {mape_ic_half_c:.2f}%")
print(f"R²: {med_r2_c:.4f} ± {r2_ic_half_c:.4f}")
print("-"*50)

# Treinar modelo final de clima com todos os dados
final_model_clima = Pipeline(steps=[
    ('preprocessor', preprocessor_clima),
    ('regressor', xgb.XGBRegressor(random_state=seed, **best_params_clima))
])
final_model_clima.fit(X_clima, y_clima)

# Salvar Modelo 2
modelo_clima_path = os.path.join(model_dir, "modelo_2_clima.pkl")
joblib.dump(final_model_clima, modelo_clima_path)

# Salvar também como modelo_final_xgb.pkl (nome esperado pelo backend)
modelo_final_path = os.path.join(model_dir, "modelo_final_xgb.pkl")
joblib.dump(final_model_clima, modelo_final_path)

print(f"\n✓ Modelo 2 salvo como '{modelo_clima_path}'")
print(f"✓ Modelo backend salvo como '{modelo_final_path}'")

# =============================================================================
# COMPARAÇÃO DOS DOIS MODELOS
# =============================================================================
print("\n" + "="*70)
print("COMPARAÇÃO: MODELO 1 vs MODELO 2")
print("="*70)
print(f"\n{'Métrica':<20} {'Modelo 1 (Sazonal)':<25} {'Modelo 2 (Clima)':<25}")
print("-"*70)
print(f"{'MAE normalizado':<20} {med_mae_s:.4f} ± {mae_ic_half_s:.4f}      {med_mae_c:.4f} ± {mae_ic_half_c:.4f}")
print(f"{'MAPE (%)':<20} {med_mape_s:.2f} ± {mape_ic_half_s:.2f}            {med_mape_c:.2f} ± {mape_ic_half_c:.2f}")
print(f"{'R²':<20} {med_r2_s:.4f} ± {r2_ic_half_s:.4f}      {med_r2_c:.4f} ± {r2_ic_half_c:.4f}")
print("-"*70)

# Calcular melhoria
melhoria_mae = ((med_mae_s - med_mae_c) / med_mae_s) * 100
melhoria_mape = ((med_mape_s - med_mape_c) / med_mape_s) * 100
melhoria_r2 = ((med_r2_c - med_r2_s) / abs(med_r2_s)) * 100

print(f"\nMelhoria do Modelo 2 sobre Modelo 1:")
print(f"  - MAE: {melhoria_mae:+.2f}%")
print(f"  - MAPE: {melhoria_mape:+.2f}%")
print(f"  - R²: {melhoria_r2:+.2f}%")

# =============================================================================
# VISUALIZAÇÕES
# =============================================================================
print("\n--- Gerando Gráficos ---")

# Gráficos Modelo 1
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(mae_scores_norm_s, bins=20, kde=True, color='blue', alpha=0.6)
plt.axvline(mae_ic_s[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(mae_ic_s[1], color='green', linestyle=':')
plt.xlabel('MAE Normalizado', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Modelo 1: Sazonalidade', fontsize=14, fontweight='bold')
plt.legend()

plt.subplot(1, 3, 2)
sns.histplot(mape_scores_s, bins=20, kde=True, color='blue', alpha=0.6)
plt.axvline(mape_ic_s[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(mape_ic_s[1], color='green', linestyle=':')
plt.xlabel('MAPE (%)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Modelo 1: Sazonalidade', fontsize=14, fontweight='bold')
plt.legend()

plt.subplot(1, 3, 3)
sns.histplot(r2_scores_s, bins=20, kde=True, color='blue', alpha=0.6)
plt.axvline(r2_ic_s[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(r2_ic_s[1], color='green', linestyle=':')
plt.xlabel('R²', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Modelo 1: Sazonalidade', fontsize=14, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.show()

# Gráficos Modelo 2
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(mae_scores_norm_c, bins=20, kde=True, color='orange', alpha=0.6)
plt.axvline(mae_ic_c[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(mae_ic_c[1], color='green', linestyle=':')
plt.xlabel('MAE Normalizado', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Modelo 2: Clima + Sazonalidade', fontsize=14, fontweight='bold')
plt.legend()

plt.subplot(1, 3, 2)
sns.histplot(mape_scores_c, bins=20, kde=True, color='orange', alpha=0.6)
plt.axvline(mape_ic_c[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(mape_ic_c[1], color='green', linestyle=':')
plt.xlabel('MAPE (%)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Modelo 2: Clima + Sazonalidade', fontsize=14, fontweight='bold')
plt.legend()

plt.subplot(1, 3, 3)
sns.histplot(r2_scores_c, bins=20, kde=True, color='orange', alpha=0.6)
plt.axvline(r2_ic_c[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(r2_ic_c[1], color='green', linestyle=':')
plt.xlabel('R²', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Modelo 2: Clima + Sazonalidade', fontsize=14, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("TREINAMENTO COMPLETO FINALIZADO!")
print("="*70)
print(f"\nArquivos salvos em: {os.path.abspath(model_dir)}")
print(f"  1. {modelo_sazonal_path} (Modelo sazonalidade)")
print(f"  2. {modelo_clima_path} (Modelo clima completo)")
print(f"  3. {modelo_final_path} (Para uso do backend)")
print("="*70)