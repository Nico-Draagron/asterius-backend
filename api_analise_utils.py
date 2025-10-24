import joblib
import numpy as np
from ClassificadorVendasSazonal import ClassificadorVendasSazonal

def prever_e_classificar(dados_entrada):
    features_modelo = [
        dados_entrada.get('mes', 1),
        dados_entrada.get('dia_semana', 0),
        dados_entrada.get('temp_media', 22.0),
        dados_entrada.get('precipitacao', 0.0),
        dados_entrada.get('radiacao', 1500.0)
    ]
    modelo = joblib.load('backend/modelo/modelo_final_xgb.pkl')
    valor_previsto = float(modelo.predict(np.array([features_modelo]))[0])
    classificador = ClassificadorVendasSazonal('backend/data/analise_sazonal_vendas.json')
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

def exportar_para_api(resultado):
    status_emoji = {
        'ALTA': '🔴',
        'MÉDIA': '🟡',
        'BAIXA': '🔵'
    }.get(resultado['classificacao'], '⚪')
    clima = resultado.get('clima', {})
    def extrair_clima(valor_cat):
        import re
        match = re.match(r"([\d\.]+)[°Cmm ]*\s*\((\w+)\)", valor_cat)
        if match:
            valor = float(match.group(1))
            categoria = match.group(2)
            return valor, categoria
        return None, None
    temp_val, temp_categoria = extrair_clima(clima.get('temperatura', ''))
    chuva_val, chuva_categoria = extrair_clima(clima.get('precipitacao', ''))
    rad_val, rad_categoria = extrair_clima(clima.get('radiacao', ''))
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
