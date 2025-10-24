

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm # Para uma barra de progresso amigável
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import joblib



# Carregar os dois datasets e adicionar coluna loja_id
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
loja1_path = os.path.join(base_dir, 'datasets', 'Loja1_processado.csv')
loja2_path = os.path.join(base_dir, 'datasets', 'Loja2_processado.csv')
loja1 = pd.read_csv(loja1_path)
loja1['loja_id'] = 1
loja2 = pd.read_csv(loja2_path)
loja2['loja_id'] = 2

# Harmonizar colunas (remover duplicatas e alinhar nomes)
loja1 = loja1.loc[:,~loja1.columns.duplicated()]
loja2 = loja2.loc[:,~loja2.columns.duplicated()]

# Alinhar colunas
common_cols = [col for col in loja1.columns if col in loja2.columns]
loja1 = loja1[common_cols]
loja2 = loja2[common_cols]

# Concatenar os dois dataframes
entrada = pd.concat([loja1, loja2], ignore_index=True)

entrada.columns

### SELEÇÂO DE VARIAVEIS

##>>>>>>MODIFUICAÇÔES NA ENTRADA<<<<<<

#>>>>>> SELEÇÂO de FEATURES


# Adicionar loja_id como feature
df=entrada[['valores','temp_max','temp_media','umid_mediana',
    'rad_max',
     'dia_semana_num','feriado',
     'day_of_year', 'mes', 'trimestre', 'Chuva_aberta', 'loja_id']]


df['dia_semana_num'] = df['dia_semana_num'].astype('category')
df['feriado'] = df['feriado'].astype('category')
df['trimestre'] = df['trimestre'].astype('category')
df['mes'] = df['mes'].astype('category')
df['loja_id'] = df['loja_id'].astype('category')


numeric_features = ['temp_max','temp_media','umid_mediana','rad_max',
    'day_of_year', 'Chuva_aberta']
categorical_features = ['dia_semana_num','feriado','trimestre','mes','loja_id']

df.describe()

# --- FASE 0: PREPARAÇÃO INICIAL ---
seed = 42

# Remoção de NA da variável alvo
df.dropna(subset=['valores'], inplace=True)

# Definir variáveis do modelo
X = df.drop('valores', axis=1)
y = df['valores']


# Definição do pré-processador com PCA após o StandardScaler
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- FASE 1: OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA ---
print("--- FASE 1: Otimização de Hiperparâmetros ---")

# Divisão única apenas para a fase de otimização
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)


def objective(trial):
    model_choice = trial.suggest_categorical('model_type', ['xgboost', 'lightgbm'])
    if model_choice == 'xgboost':
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True)
        }
        regressor = xgb.XGBRegressor(random_state=seed, **params)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True)
        }
        regressor = lgb.LGBMRegressor(random_state=seed, **params)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    score = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    return score.mean()

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
study.optimize(objective, n_trials=50) # Mantenha um número razoável de trials

best_params = study.best_params
print(f"\nMelhores Hiperparâmetros encontrados: {best_params}")


# --- FASE 2: AVALIAÇÃO ROBUSTA COM BOOTSTRAP ---
print("\n--- FASE 2: Avaliação Robusta com Bootstrap  ---")

n_bootstrap_iterations = 100
n_samples = len(df)
indices = np.arange(n_samples) # Array de 0 a n-1

mae_scores, rmse_scores, r2_scores, mape_scores = [], [], [], []

for i in tqdm(range(n_bootstrap_iterations), desc="Executando Bootstrap "):
    iteration_seed = seed + i

    # Ele realiza a amostragem com reposição e estratificada de uma só vez.
    in_bag_indices = resample(
        indices,                # Amostrar a partir dos índices
        replace=True,           # Bootstrap (amostragem com reposição)
        n_samples=n_samples,  # O tamanho da amostra de bootstrap é o mesmo do original
      random_state=iteration_seed
    )

    # Out-of-Bag
    out_of_bag_indices = np.setdiff1d(indices, np.unique(in_bag_indices))

    if len(out_of_bag_indices) == 0:
        continue

    X_train_boot, y_train_boot = X.iloc[in_bag_indices], y.iloc[in_bag_indices]
    X_test_boot, y_test_boot = X.iloc[out_of_bag_indices], y.iloc[out_of_bag_indices]

    # O resto do loop (treino, predição, armazenamento de métricas) é idêntico
    model_pipeline_boot = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=seed, **best_params))
    ])

    model_pipeline_boot.fit(X_train_boot, y_train_boot)
    y_pred_boot = model_pipeline_boot.predict(X_test_boot)

    # Calcular MAE, RMSE, R² e MAPE
    mae_scores.append(mean_absolute_error(y_test_boot, y_pred_boot))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_boot, y_pred_boot)))
    r2_scores.append(r2_score(y_test_boot, y_pred_boot))
    
    # Calcular MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_boot - y_pred_boot) / y_test_boot)) * 100
    mape_scores.append(mape)


# --- FASE 3: ANÁLISE DOS RESULTADOS DO BOOTSTRAP ---

# --- NOVA ANÁLISE: Mediana, IC/2, MAE normalizado, sem títulos nos gráficos ---
mae_scores_norm = np.array(mae_scores) / y.max()

def ic_halfwidth(arr):
    ic = np.percentile(arr, [2.5, 97.5])
    return (ic[1] - ic[0]) / 2, ic

med_mae = np.median(mae_scores_norm)
mae_ic_half, mae_ic = ic_halfwidth(mae_scores_norm)

med_mape = np.median(mape_scores)
mape_ic_half, mape_ic = ic_halfwidth(mape_scores)

med_r2 = np.median(r2_scores)
r2_ic_half, r2_ic = ic_halfwidth(r2_scores)

print("-" * 50)
print(f"MAE normalizado: {med_mae:.2f} ± {mae_ic_half:.2f}")
print(f"MAPE: {med_mape:.2f}% ± {mape_ic_half:.2f}%")
print(f"R²: {med_r2:.2f} ± {r2_ic_half:.2f}")
print("-" * 50)

# Visualização das distribuições (sem título, sem marcação de média/mediana, só IC)
plt.figure(figsize=(10, 6))
sns.histplot(mae_scores_norm, bins=20, kde=True)
plt.axvline(mae_ic[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(mae_ic[1], color='green', linestyle=':', label=None)
plt.xlabel('MAE Normalizado (0-1)', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
## Linhas antigas de visualização removidas (usamos apenas as novas visualizações baseadas em mediana e IC)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualização da distribuição da métrica R²

plt.figure(figsize=(10, 6))
sns.histplot(r2_scores, bins=20, kde=True)
plt.axvline(r2_ic[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(r2_ic[1], color='green', linestyle=':', label=None)
plt.xlabel('Coeficiente de Determinação (R²)', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualização da distribuição da métrica MAPE

plt.figure(figsize=(10, 6))
sns.histplot(mape_scores, bins=20, kde=True)
plt.axvline(mape_ic[0], color='green', linestyle=':', label='IC 95%')
plt.axvline(mape_ic[1], color='green', linestyle=':', label=None)
plt.xlabel('Mean Absolute Percentage Error (MAPE) %', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- FASE FINAL: SALVAR O MODELO TREINADO EM .pkl ---
print("\n--- FASE FINAL: Salvando o modelo treinado em .pkl ---")
# Treina o modelo final com todos os dados
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(random_state=seed, **best_params))
])
final_model_pipeline.fit(X, y)
# Salva o pipeline completo (pré-processamento + modelo) na pasta modelo
import os
modelo_path = os.path.join("modelo", "modelo_teste_xgb.pkl")
joblib.dump(final_model_pipeline, modelo_path)
print(f"Modelo salvo como '{modelo_path}'. Pronto para exportação e uso!")
