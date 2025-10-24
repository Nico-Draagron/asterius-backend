from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
def exportar_para_api(resultado):
    try:
        status_emoji = {
            'ALTA': '🔴',
            'MÉDIA': '🟡',
            'BAIXA': '🔵'
        }.get(resultado['classificacao'], '⚪')
        clima = resultado.get('clima', {})
        temp_cat = clima.get('temperatura', '')
        chuva_cat = clima.get('precipitacao', '')
        rad_cat = clima.get('radiacao', '')
        # Extrai categoria e valor
        def extrair_clima(valor_cat):
            import re
            match = re.match(r"([\d\.]+)[°Cmm ]*\s*\((\w+)\)", valor_cat)
            if match:
                valor = float(match.group(1))
                categoria = match.group(2)
                return valor, categoria
            return None, None
        temp_val, temp_categoria = extrair_clima(temp_cat)
        chuva_val, chuva_categoria = extrair_clima(chuva_cat)
        rad_val, rad_categoria = extrair_clima(rad_cat)
        clima_json = {
            "temperatura": {"valor": temp_val, "categoria": temp_categoria, "emoji": "🌡️"},
            "precipitacao": {"valor": chuva_val, "categoria": chuva_categoria, "emoji": "🌧️"},
            "radiacao": {"valor": rad_val, "categoria": rad_categoria, "emoji": "☀️"}
        }
        return {
            "previsao": {
                "valor": resultado.get("valor_previsto"),
                "classificacao": resultado.get("classificacao"),
                "status_emoji": status_emoji,
                "desvio_percentual": resultado.get("desvio_percentual")
            },
            "contexto": {
                "mes": resultado.get("mes"),
                "dia": resultado.get("dia"),
                "baseline": resultado.get("baseline_dia", None),
                "venda_esperada": resultado.get("venda_esperada"),
                "impacto_climatico": resultado.get("impacto_climatico_total")
            },
            "clima": clima_json,
            "fatores": {
                "positivos": resultado.get("fatores_positivos", []),
                "negativos": resultado.get("fatores_negativos", [])
            },
            "thresholds": {
                "baixa": resultado.get("threshold_baixa"),
                "alta": resultado.get("threshold_alta")
            }
        }
    except Exception as e:
        return {"error": str(e)}
app = FastAPI()

@app.post("/api/analise")
async def api_analise(request: Request):
    try:
        dados = await request.json()
        resultado = prever_e_classificar(dados)
        api_json = exportar_para_api(resultado)
        return JSONResponse(content=api_json)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run("predicao_com_classificacao:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # ...existing code...
import joblib
import os
from ClassificadorVendasSazonal import ClassificadorVendasSazonal
import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)

# Caminhos
CAMINHO_MODELO = 'backend/modelo/modelo_final_xgb.pkl'
CAMINHO_JSON = 'backend/data/analise_sazonal_vendas.json'

# Carregar modelo XGBoost
modelo = joblib.load(CAMINHO_MODELO)
# Carregar classificador sazonal
classificador = ClassificadorVendasSazonal(CAMINHO_JSON)

def prever_e_classificar(dados_entrada):
    # Features para o modelo (ajuste conforme necessário)
    features_modelo = [
        dados_entrada.get('mes', 1),
        dados_entrada.get('dia_semana', 0),
        dados_entrada.get('temp_media', 22.0),
        dados_entrada.get('precipitacao', 0.0),
        dados_entrada.get('radiacao', 1500.0)
        # Adicione outras features conforme o modelo
    ]
    # Previsão
    valor_previsto = float(modelo.predict(np.array([features_modelo]))[0])
    # Classificação sazonal completa
    analise = classificador.classificar_completo_sazonal(
        valor_previsto,
        dados_entrada.get('mes', 1),
        dados_entrada.get('dia_semana', 0),
        dados_entrada.get('temp_media', 22.0),
        dados_entrada.get('precipitacao', 0.0),
        dados_entrada.get('radiacao', 1500.0)
    )
    resultado = {'previsao': valor_previsto, **analise}
    return resultado

def print_resultado(resultado):
    cor = {
        'ALTA': Fore.RED,
        'MÉDIA': Fore.YELLOW,
        'BAIXA': Fore.BLUE
    }.get(resultado['classificacao'], Fore.WHITE)
    print(f"{cor}🎯 {resultado['dia']} de {resultado['mes']}{Style.RESET_ALL}")
    print(f"  Previsão: R$ {resultado['previsao']:,.2f}")
    print(f"  Esperado (ajustado): R$ {resultado['venda_esperada']:,.2f}")
    print(f"  Classificação: {cor}{resultado['classificacao']}{Style.RESET_ALL}")
    print(f"  Desvio: {resultado['desvio_percentual']:+.1f}%")
    print(f"  Thresholds: Baixa < R$ {resultado['threshold_baixa']:.2f} | Alta > R$ {resultado['threshold_alta']:.2f}")
    print(f"  Mediana histórica: R$ {resultado['mediana']:.2f} | Registros: {resultado['count']}")
    print(f"  Impacto climático total: ×{resultado['impacto_climatico_total']}")
    print(f"  Clima: {resultado['clima']}")
    print(f"  {resultado['contexto']}")
    if resultado['fatores_positivos']:
        print(Fore.GREEN + "  ✅ Fatores Positivos:")
        for fator in resultado['fatores_positivos']:
            print(Fore.GREEN + f"     • {fator}")
    if resultado['fatores_negativos']:
        print(Fore.CYAN + "  ❌ Fatores Negativos:")
        for fator in resultado['fatores_negativos']:
            print(Fore.CYAN + f"     • {fator}")
    print(Style.RESET_ALL)

if __name__ == "__main__":
    print(Fore.MAGENTA + "="*70)
    print(Fore.MAGENTA + "🔮 EXEMPLO 1: Previsão Única")
    print(Fore.MAGENTA + "="*70)
    entrada = {
        'mes': 10,
        'dia_semana': 6,
        'temp_media': 26,
        'precipitacao': 0,
        'radiacao': 1800
    }
    resultado = prever_e_classificar(entrada)
    print_resultado(resultado)

    print(Fore.MAGENTA + "="*70)
    print(Fore.MAGENTA + "📅 EXEMPLO 2: Semana Completa")
    print(Fore.MAGENTA + "="*70)
    semana = [
        {'mes': 10, 'dia_semana': i, 'temp_media': 25+i, 'precipitacao': i*2, 'radiacao': 1200+i*100} for i in range(7)
    ]
    for entrada in semana:
        resultado = prever_e_classificar(entrada)
        print_resultado(resultado)

    print(Fore.MAGENTA + "="*70)
    print(Fore.MAGENTA + "🌦️ EXEMPLO 3: Comparação de Cenários")
    print(Fore.MAGENTA + "="*70)
    # Clima bom
    entrada_bom = {
        'mes': 10, 'dia_semana': 6, 'temp_media': 27, 'precipitacao': 0, 'radiacao': 2000
    }
    resultado_bom = prever_e_classificar(entrada_bom)
    print(Fore.GREEN + "Cenário BOM:")
    print_resultado(resultado_bom)
    # Clima ruim
    entrada_ruim = {
        'mes': 10, 'dia_semana': 6, 'temp_media': 15, 'precipitacao': 35, 'radiacao': 300
    }
    resultado_ruim = prever_e_classificar(entrada_ruim)
    print(Fore.RED + "Cenário RUIM:")
    print_resultado(resultado_ruim)
