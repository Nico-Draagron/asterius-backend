import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
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

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================
seed = 42
np.random.seed(seed)

print("="*80)
print("MODELO OTIMIZADO COM BOAS PRÁTICAS DE ML")
print("="*80)

# =============================================================================
# CARREGAR E CONCATENAR DATASETS
# =============================================================================
print("\n--- Carregando Datasets ---")
loja1 = pd.read_csv('datasets/Loja1_processado.csv')
loja1['loja_id'] = 1
loja2 = pd.read_csv('datasets/Loja2_processado.csv')
loja2['loja_id'] = 2

# Harmonizar colunas
loja1 = loja1.loc[:, ~loja1.columns.duplicated()]
loja2 = loja2.loc[:, ~loja2.columns.duplicated()]

common_cols = [col for col in loja1.columns if col in loja2.columns]
loja1 = loja1[common_cols]
loja2 = loja2[common_cols]

# Concatenar
entrada = pd.concat([loja1, loja2], ignore_index=True)
print(f"✓ Dataset concatenado: {entrada.shape[0]} linhas, {entrada.shape[1]} colunas")

# =============================================================================
# PREPROCESSAMENTO E FEATURE ENGINEERING
# =============================================================================
print("\n--- Feature Engineering ---")

# Converter data para datetime e ordenar cronologicamente
entrada['data'] = pd.to_datetime(entrada['data'])
entrada = entrada.sort_values('data').reset_index(drop=True)
print("✓ Dados ordenados cronologicamente")

# Garantir valores numéricos
for col in ['mes', 'day_of_year', 'dia_mes', 'semana_ano']:
    if col in entrada.columns:
        entrada[col] = pd.to_numeric(entrada[col], errors='coerce').fillna(1).astype(int)

# 1. CRIAR FEATURES CÍCLICAS TEMPORAIS
print("\n[1] Criando features cíclicas...")

# Mês (ciclo de 12)
entrada['mes_sin'] = np.sin(2 * np.pi * entrada['mes'] / 12)
entrada['mes_cos'] = np.cos(2 * np.pi * entrada['mes'] / 12)
print("  ✓ mes_sin, mes_cos")

# Dia do ano (ciclo de 365)
entrada['day_sin'] = np.sin(2 * np.pi * entrada['day_of_year'] / 365)
entrada['day_cos'] = np.cos(2 * np.pi * entrada['day_of_year'] / 365)
print("  ✓ day_sin, day_cos")

# Semana do ano (ciclo de 52)
entrada['semana_ano_sin'] = np.sin(2 * np.pi * entrada['semana_ano'] / 52)
entrada['semana_ano_cos'] = np.cos(2 * np.pi * entrada['semana_ano'] / 52)
print("  ✓ semana_ano_sin, semana_ano_cos")

# 2. CRIAR FEATURES BINÁRIAS
print("\n[2] Criando features binárias...")

# Fim de semana (sábado=5, domingo=6)
entrada['fim_de_semana'] = (entrada['dia_semana_num'].isin([5, 6])).astype(int)
print("  ✓ fim_de_semana")

# Início do mês (dias 1-5)
entrada['inicio_mes'] = (entrada['dia_mes'] <= 5).astype(int)
print("  ✓ inicio_mes")

# Fim do mês (dias 26-31)
entrada['fim_mes'] = (entrada['dia_mes'] >= 26).astype(int)
print("  ✓ fim_mes")

# 3. SELECIONAR FEATURES FINAIS (removendo redundâncias)
print("\n[3] Selecionando features finais...")

# Features numéricas: clima + cíclicas
numeric_features = [
    # Clima
    'temp_media', 'umid_mediana', 'rad_max', 'precipitacao_total', 'vento_vel_media',
    # Temporais cíclicas
    'mes_sin', 'mes_cos', 'day_sin', 'day_cos', 'semana_ano_sin', 'semana_ano_cos'
]

# Features categóricas
categorical_features = [
    'dia_semana_num', 'feriado', 'vespera_feriado', 'loja_id',
    'fim_de_semana', 'inicio_mes', 'fim_mes'
]

# Criar dataset final
selected_cols = ['data', 'valores'] + numeric_features + categorical_features
df = entrada[selected_cols].copy()

# Remover NaN na variável alvo
df.dropna(subset=['valores'], inplace=True)

print(f"\n✓ Features selecionadas:")
print(f"  - Numéricas: {len(numeric_features)}")
print(f"  - Categóricas: {len(categorical_features)}")
print(f"  - Total: {len(numeric_features) + len(categorical_features)}")

# Converter categóricas
for col in categorical_features:
    df[col] = df[col].astype('category')

# =============================================================================
# VALIDAÇÃO TEMPORAL - SPLIT POR DATA
# =============================================================================
print("\n--- Validação Temporal ---")

# Ordenar por data e criar split temporal (80% treino, 20% teste)
df = df.sort_values('data').reset_index(drop=True)
split_idx = int(0.8 * len(df))

data_split = df['data'].iloc[split_idx]
print(f"✓ Split temporal em: {data_split.strftime('%Y-%m-%d')}")
print(f"  - Treino: {split_idx} observações (até {data_split.strftime('%Y-%m-%d')})")
print(f"  - Teste: {len(df) - split_idx} observações (após {data_split.strftime('%Y-%m-%d')})")

# Separar em treino e teste TEMPORAL
train_data = df.iloc[:split_idx].copy()
test_data = df.iloc[split_idx:].copy()

X_train = train_data.drop(['data', 'valores'], axis=1)
y_train = train_data['valores']
X_test = test_data.drop(['data', 'valores'], axis=1)
y_test = test_data['valores']

# Dataset completo para bootstrap final
X = df.drop(['data', 'valores'], axis=1)
y = df['valores']

# =============================================================================
# DEFINIR PRÉ-PROCESSADOR
# =============================================================================
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# =============================================================================
# FASE 1: OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA (Time Series CV)
# =============================================================================
print("\n" + "="*80)
print("FASE 1: OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*80)

# Time Series Cross-Validation (5 splits)
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=seed, **params))
    ])
    
    # Cross-validation temporal
    mae_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_pipeline.fit(X_tr, y_tr)
        y_pred = model_pipeline.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    return -np.mean(mae_scores)  # Negativo porque Optuna maximiza

print("Iniciando otimização com Time Series Cross-Validation...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(f"\n✓ Melhores Hiperparâmetros encontrados:")
for param, value in best_params.items():
    print(f"  - {param}: {value}")

# =============================================================================
# FASE 2: AVALIAÇÃO NO CONJUNTO DE TESTE TEMPORAL
# =============================================================================
print("\n" + "="*80)
print("FASE 2: AVALIAÇÃO NO TESTE TEMPORAL")
print("="*80)

# Treinar com melhores hiperparâmetros
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(random_state=seed, **best_params))
])

final_model.fit(X_train, y_train)
y_pred_test = final_model.predict(X_test)

# Métricas no teste
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
mae_norm_test = mae_test / y_test.max()

print("\n" + "-"*60)
print("RESULTADOS NO TESTE TEMPORAL (dados futuros)")
print("-"*60)
print(f"MAE normalizado: {mae_norm_test:.4f}")
print(f"MAE absoluto: {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MAPE: {mape_test:.2f}%")
print(f"R²: {r2_test:.4f}")
print("-"*60)

# =============================================================================
# FASE 3: AVALIAÇÃO ROBUSTA COM BOOTSTRAP (dados de treino)
# =============================================================================
print("\n" + "="*80)
print("FASE 3: AVALIAÇÃO ROBUSTA COM BOOTSTRAP")
print("="*80)

n_bootstrap_iterations = 100
n_samples = len(X_train)
indices = np.arange(n_samples)

mae_scores, rmse_scores, r2_scores, mape_scores = [], [], [], []

for i in tqdm(range(n_bootstrap_iterations), desc="Executando Bootstrap"):
    iteration_seed = seed + i
    
    in_bag_indices = resample(indices, replace=True, n_samples=n_samples, random_state=iteration_seed)
    out_of_bag_indices = np.setdiff1d(indices, np.unique(in_bag_indices))
    
    if len(out_of_bag_indices) == 0:
        continue
    
    X_train_boot = X_train.iloc[in_bag_indices]
    y_train_boot = y_train.iloc[in_bag_indices]
    X_test_boot = X_train.iloc[out_of_bag_indices]
    y_test_boot = y_train.iloc[out_of_bag_indices]
    
    model_pipeline_boot = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=seed, **best_params))
    ])
    
    model_pipeline_boot.fit(X_train_boot, y_train_boot)
    y_pred_boot = model_pipeline_boot.predict(X_test_boot)
    
    mae_scores.append(mean_absolute_error(y_test_boot, y_pred_boot))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_boot, y_pred_boot)))
    r2_scores.append(r2_score(y_test_boot, y_pred_boot))
    mape_scores.append(np.mean(np.abs((y_test_boot - y_pred_boot) / y_test_boot)) * 100)

# Análise dos resultados do Bootstrap
mae_scores_norm = np.array(mae_scores) / y_train.max()

def ic_halfwidth(arr):
    ic = np.percentile(arr, [2.5, 97.5])
    return (ic[1] - ic[0]) / 2, ic

med_mae = np.median(mae_scores_norm)
mae_ic_half, mae_ic = ic_halfwidth(mae_scores_norm)

med_mape = np.median(mape_scores)
mape_ic_half, mape_ic = ic_halfwidth(mape_scores)

med_r2 = np.median(r2_scores)
r2_ic_half, r2_ic = ic_halfwidth(r2_scores)

print("\n" + "-"*60)
print("RESULTADOS BOOTSTRAP (validação cruzada no treino)")
print("-"*60)
print(f"MAE normalizado: {med_mae:.4f} ± {mae_ic_half:.4f}")
print(f"MAPE: {med_mape:.2f}% ± {mape_ic_half:.2f}%")
print(f"R²: {med_r2:.4f} ± {r2_ic_half:.4f}")
print("-"*60)

# =============================================================================
# VISUALIZAÇÕES
# =============================================================================
print("\n--- Gerando Gráficos ---")

# 1. Distribuições do Bootstrap
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MAE
axes[0].hist(mae_scores_norm, bins=20, alpha=0.7, edgecolor='black')
axes[0].axvline(mae_ic[0], color='green', linestyle=':', linewidth=2, label='IC 95%')
axes[0].axvline(mae_ic[1], color='green', linestyle=':', linewidth=2)
axes[0].set_xlabel('MAE Normalizado (0-1)', fontsize=12)
axes[0].set_ylabel('Frequência', fontsize=12)
axes[0].set_title('Distribuição MAE (Bootstrap)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAPE
axes[1].hist(mape_scores, bins=20, alpha=0.7, edgecolor='black', color='orange')
axes[1].axvline(mape_ic[0], color='green', linestyle=':', linewidth=2, label='IC 95%')
axes[1].axvline(mape_ic[1], color='green', linestyle=':', linewidth=2)
axes[1].set_xlabel('MAPE (%)', fontsize=12)
axes[1].set_ylabel('Frequência', fontsize=12)
axes[1].set_title('Distribuição MAPE (Bootstrap)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# R²
axes[2].hist(r2_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
axes[2].axvline(r2_ic[0], color='red', linestyle=':', linewidth=2, label='IC 95%')
axes[2].axvline(r2_ic[1], color='red', linestyle=':', linewidth=2)
axes[2].set_xlabel('R²', fontsize=12)
axes[2].set_ylabel('Frequência', fontsize=12)
axes[2].set_title('Distribuição R² (Bootstrap)', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Predito vs Real (Teste Temporal)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, edgecolor='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predição Perfeita')
plt.xlabel('Valores Reais', fontsize=14)
plt.ylabel('Valores Preditos', fontsize=14)
plt.title('Predição vs Real (Teste Temporal)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Resíduos ao longo do tempo (Teste)
residuos = y_test.values - y_pred_test
plt.figure(figsize=(12, 5))
plt.plot(test_data['data'].values, residuos, alpha=0.6, marker='o', linestyle='-', markersize=3)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Resíduos (Real - Predito)', fontsize=14)
plt.title('Resíduos ao Longo do Tempo (Teste Temporal)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# FASE FINAL: TREINAR MODELO COM TODOS OS DADOS E SALVAR
# =============================================================================
print("\n" + "="*80)
print("FASE FINAL: SALVANDO MODELO")
print("="*80)

# Treinar com TODOS os dados
final_model_all = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(random_state=seed, **best_params))
])
final_model_all.fit(X, y)

# Salvar modelo
os.makedirs("modelo", exist_ok=True)
modelo_path = os.path.join("modelo", "modelo_otimizado_xgb.pkl")
joblib.dump(final_model_all, modelo_path)

print(f"\n✓ Modelo salvo como '{modelo_path}'")
print(f"✓ Features utilizadas: {len(numeric_features) + len(categorical_features)}")
print(f"✓ Observações de treino: {len(y)}")

# Salvar resumo dos resultados
resumo = {
    'teste_temporal': {
        'mae_normalizado': mae_norm_test,
        'mape': mape_test,
        'r2': r2_test
    },
    'bootstrap': {
        'mae_normalizado': med_mae,
        'mape': med_mape,
        'r2': med_r2
    },
    'best_params': best_params,
    'features': {
        'numericas': numeric_features,
        'categoricas': categorical_features
    }
}

resumo_path = os.path.join("modelo", "resumo_modelo.txt")
with open(resumo_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RESUMO DO MODELO OTIMIZADO\n")
    f.write("="*80 + "\n\n")
    f.write("TESTE TEMPORAL (dados futuros):\n")
    f.write(f"  MAE normalizado: {mae_norm_test:.4f}\n")
    f.write(f"  MAPE: {mape_test:.2f}%\n")
    f.write(f"  R²: {r2_test:.4f}\n\n")
    f.write("BOOTSTRAP (validação cruzada):\n")
    f.write(f"  MAE normalizado: {med_mae:.4f} ± {mae_ic_half:.4f}\n")
    f.write(f"  MAPE: {med_mape:.2f}% ± {mape_ic_half:.2f}%\n")
    f.write(f"  R²: {med_r2:.4f} ± {r2_ic_half:.4f}\n\n")
    f.write("HIPERPARÂMETROS:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")

print(f"✓ Resumo salvo em '{resumo_path}'")

print("\n" + "="*80)
print("TREINAMENTO COMPLETO FINALIZADO!")
print("="*80)
print("\nMELHORIAS IMPLEMENTADAS:")
print("  ✓ Features cíclicas (mes, day, semana)")
print("  ✓ Features binárias (fim_de_semana, inicio_mes, fim_mes)")
print("  ✓ Validação temporal (split por data)")
print("  ✓ Time Series Cross-Validation no Optuna")
print("  ✓ Regularização L1/L2 adicionada")
print("  ✓ Remoção de multicolinearidade")
print("  ✓ Avaliação em dados futuros (teste temporal)")