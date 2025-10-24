import json
import os

class ClassificadorVendasSazonal:
    def _classificar_temperatura(self, temp_media):
        if temp_media < 18:
            return 'fria', 0.56
        elif temp_media < 24:
            return 'amena', 1.0
        elif temp_media < 28:
            return 'quente', 1.28
        else:
            return 'muito_quente', 1.38

    def _classificar_chuva(self, precipitacao):
        if precipitacao == 0:
            return 'sem_chuva', 1.0
        elif precipitacao <= 10:
            return 'leve', 0.67
        elif precipitacao <= 30:
            return 'media', 0.59
        else:
            return 'forte', 0.51

    def _classificar_radiacao(self, radiacao):
        if radiacao < 500:
            return 'baixa', 0.45
        elif radiacao < 1500:
            return 'media', 0.66
        else:
            return 'alta', 1.0

    def classificar_completo_sazonal(self, valor_previsto, mes, dia_semana, temp_media, precipitacao, radiacao):
        # Validações
        if not (1 <= mes <= 12):
            raise ValueError('mes deve ser entre 1 e 12')
        if not (0 <= dia_semana <= 6):
            raise ValueError('dia_semana deve ser entre 0 (Segunda) e 6 (Domingo)')
        mes_str = str(mes)
        dia_str = str(dia_semana)
        try:
            info_mes = self.dados[mes_str]
            info_dia = info_mes['dias'][dia_str]
        except KeyError:
            raise ValueError('Dados históricos não encontrados para o mês/dia informado.')
        baseline = info_dia.get('media', info_dia.get('baseline', 0))
        q1 = info_dia['q1']
        q3 = info_dia['q3']
        mediana = info_dia['mediana']
        count = info_dia['count']
        nome_mes = info_mes['nome']
        nome_dia = info_dia['nome']

        # Classificação climática
        cat_temp, mult_temp = self._classificar_temperatura(temp_media)
        cat_chuva, mult_chuva = self._classificar_chuva(precipitacao)
        cat_rad, mult_rad = self._classificar_radiacao(radiacao)
        impacto_climatico_total = round(mult_temp * mult_chuva * mult_rad, 2)

        # Venda esperada ajustada
        venda_esperada = baseline * impacto_climatico_total
        threshold_baixa = q1 * impacto_climatico_total
        threshold_alta = q3 * impacto_climatico_total

        # Classificação
        if valor_previsto < threshold_baixa:
            classificacao = 'BAIXA'
            contexto = f"A previsão está abaixo do esperado para {nome_dia} de {nome_mes} considerando clima."
        elif valor_previsto > threshold_alta:
            classificacao = 'ALTA'
            contexto = f"A previsão está acima do esperado para {nome_dia} de {nome_mes} considerando clima."
        else:
            classificacao = 'MÉDIA'
            contexto = f"A previsão está dentro do intervalo típico para {nome_dia} de {nome_mes} considerando clima."

        # Desvio percentual
        desvio_percentual = round(((valor_previsto - venda_esperada) / venda_esperada) * 100, 1) if venda_esperada else 0.0

        # Fatores positivos/negativos
        fatores_positivos = []
        fatores_negativos = []
        # Temperatura
        if cat_temp == 'muito_quente':
            fatores_positivos.append("🌡️ Calor forte impulsiona vendas (+38%)")
        elif cat_temp == 'quente':
            fatores_positivos.append("🌡️ Temperatura ideal para vendas (+28%)")
        elif cat_temp == 'amena':
            fatores_positivos.append("🌡️ Temperatura neutra (sem impacto)")
        elif cat_temp == 'fria':
            fatores_negativos.append("🌡️ Temperatura fria reduz vendas (-44%)")
        # Chuva
        if cat_chuva == 'sem_chuva':
            fatores_positivos.append("☀️ Sem chuva - condição ideal")
        elif cat_chuva == 'leve':
            fatores_negativos.append(f"🌧️ Chuva leve ({precipitacao:.1f}mm) reduz vendas (-33%)")
        elif cat_chuva == 'media':
            fatores_negativos.append(f"🌧️ Chuva moderada ({precipitacao:.1f}mm) prejudica muito (-41%)")
        elif cat_chuva == 'forte':
            fatores_negativos.append(f"⛈️ Chuva forte ({precipitacao:.1f}mm) impacto severo (-49%)")
        # Radiação
        if cat_rad == 'alta':
            fatores_positivos.append(f"☀️ Sol forte (rad {radiacao:.0f}) - melhor cenário")
        elif cat_rad == 'media':
            fatores_negativos.append(f"⛅ Parcialmente nublado (rad {radiacao:.0f}) reduz vendas (-34%)")
        elif cat_rad == 'baixa':
            fatores_negativos.append(f"☁️ Muito nublado (rad {radiacao:.0f}) impacto negativo (-55%)")

        return {
            'classificacao': classificacao,
            'valor_previsto': float(valor_previsto),
            'mes': nome_mes,
            'dia': nome_dia,
            'threshold_baixa': round(threshold_baixa, 2),
            'threshold_alta': round(threshold_alta, 2),
            'mediana': mediana,
            'count': count,
            'contexto': contexto,
            'venda_esperada': round(venda_esperada, 2),
            'desvio_percentual': desvio_percentual,
            'fatores_positivos': fatores_positivos,
            'fatores_negativos': fatores_negativos,
            'impacto_climatico_total': impacto_climatico_total,
            'clima': {
                'temperatura': f"{temp_media}°C ({cat_temp})",
                'precipitacao': f"{precipitacao}mm ({cat_chuva})",
                'radiacao': f"{radiacao} ({cat_rad})"
            },
            'thresholds_ajustados': {
                'baixa': round(threshold_baixa, 2),
                'alta': round(threshold_alta, 2)
            }
        }
    """
    Classifica previsões de vendas usando thresholds sazonais por mês e dia da semana.
    """
    MESES_PT = [
        '', 'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
        'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
    ]
    DIAS_PT = [
        'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'
    ]

    def __init__(self, caminho_json='backend/data/analise_sazonal_vendas.json'):
        if not os.path.exists(caminho_json):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_json}")
        with open(caminho_json, 'r', encoding='utf-8') as f:
            self.dados = json.load(f)

    def classificar_por_mes_e_dia(self, valor_previsto, mes, dia_semana):
        # Validações
        if not (1 <= mes <= 12):
            raise ValueError('mes deve ser entre 1 e 12')
        if not (0 <= dia_semana <= 6):
            raise ValueError('dia_semana deve ser entre 0 (Segunda) e 6 (Domingo)')
        mes_str = str(mes)
        dia_str = str(dia_semana)
        try:
            info_mes = self.dados[mes_str]
            info_dia = info_mes['dias'][dia_str]
        except KeyError:
            raise ValueError('Dados históricos não encontrados para o mês/dia informado.')
        q1 = info_dia['q1']
        q3 = info_dia['q3']
        mediana = info_dia['mediana']
        count = info_dia['count']
        nome_mes = info_mes['nome']
        nome_dia = info_dia['nome']
        # Classificação
        if valor_previsto < q1:
            classificacao = 'BAIXA'
            contexto = f"A previsão está abaixo do histórico para {nome_dia} de {nome_mes} (Q1={q1:.2f})."
        elif valor_previsto > q3:
            classificacao = 'ALTA'
            contexto = f"A previsão está acima do histórico para {nome_dia} de {nome_mes} (Q3={q3:.2f})."
        else:
            classificacao = 'MÉDIA'
            contexto = f"A previsão está dentro do intervalo típico para {nome_dia} de {nome_mes} (Q1={q1:.2f} a Q3={q3:.2f})."
        return {
            'classificacao': classificacao,
            'valor_previsto': float(valor_previsto),
            'mes': nome_mes,
            'dia': nome_dia,
            'threshold_baixa': q1,
            'threshold_alta': q3,
            'mediana': mediana,
            'count': count,
            'contexto': contexto
        }
