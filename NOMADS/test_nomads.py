#!/usr/bin/env python3
"""
Teste simples do NOMADS - Verificar se consegue baixar dados
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import os

async def test_nomads_simple():
    """Teste básico de conexão com NOMADS"""
    print("🌍 Testando conexão com NOMADS...")
    
    # URL base do NOMADS
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    
    # Data de hoje
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    
    # Parâmetros básicos para um arquivo pequeno
    params = {
        "file": "gfs.t06z.pgrb2.0p25.f001",
        "dir": f"/gfs.{date_str}/06/atmos",
        "subregion": "",
        "leftlon": "-50",
        "rightlon": "-40", 
        "toplat": "-10",
        "bottomlat": "-20",
        "var_TMP": "on",
        "lev_2_m_above_ground": "on"
    }
    
    print(f"📅 Tentando baixar dados de: {date_str}")
    print(f"🔗 URL: {base_url}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession() as session:
            print("🔄 Fazendo requisição...")
            async with session.get(base_url, params=params, timeout=timeout) as response:
                print(f"📊 Status: {response.status}")
                print(f"📏 Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.read()
                    print(f"✅ Sucesso! Dados baixados: {len(data)} bytes")
                    
                    # Salva arquivo de teste
                    test_file = f"dados/cache/test_gfs_{date_str}.grib2"
                    os.makedirs("dados/cache", exist_ok=True)
                    
                    with open(test_file, 'wb') as f:
                        f.write(data)
                    print(f"💾 Arquivo salvo: {test_file}")
                    
                    return True
                else:
                    print(f"❌ Erro HTTP: {response.status}")
                    text = await response.text()
                    print(f"📄 Resposta: {text[:500]}...")
                    return False
                    
    except asyncio.TimeoutError:
        print("❌ Timeout - NOMADS pode estar lento")
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

async def test_different_dates():
    """Testa diferentes datas para encontrar dados disponíveis"""
    print("\n🗓️ Testando diferentes datas...")
    
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    
    # Testa últimos 3 dias
    for i in range(3):
        test_date = datetime.now() - timedelta(days=i)
        date_str = test_date.strftime("%Y%m%d")
        
        for cycle in ["00", "06", "12", "18"]:
            params = {
                "file": f"gfs.t{cycle}z.pgrb2.0p25.f003",
                "dir": f"/gfs.{date_str}/{cycle}/atmos",
                "subregion": "",
                "leftlon": "-50",
                "rightlon": "-40",
                "toplat": "-10", 
                "bottomlat": "-20",
                "var_TMP": "on",
                "lev_2_m_above_ground": "on"
            }
            
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession() as session:
                    async with session.get(base_url, params=params, timeout=timeout) as response:
                        if response.status == 200:
                            print(f"✅ Disponível: {date_str} ciclo {cycle}Z")
                            return date_str, cycle
                        else:
                            print(f"❌ {date_str} {cycle}Z: HTTP {response.status}")
            except:
                print(f"❌ {date_str} {cycle}Z: Erro de conexão")
    
    return None, None

async def main():
    """Função principal de teste"""
    print("🚀 TESTE NOMADS - ASTERIUS")
    print("=" * 40)
    
    # Cria diretórios
    os.makedirs("dados/nomads", exist_ok=True)
    os.makedirs("dados/cache", exist_ok=True)
    os.makedirs("dados/logs", exist_ok=True)
    
    # Teste 1: Conexão básica
    success = await test_nomads_simple()
    
    if not success:
        print("\n🔍 Teste básico falhou. Tentando outras datas...")
        # Teste 2: Diferentes datas
        date, cycle = await test_different_dates()
        
        if date and cycle:
            print(f"\n✅ Encontrados dados disponíveis: {date} {cycle}Z")
        else:
            print("\n❌ Nenhum dado disponível encontrado")
            print("\n💡 Possíveis problemas:")
            print("1. NOMADS pode estar em manutenção")
            print("2. Dados ainda não foram processados")
            print("3. Problema de conectividade")
            print("4. Parâmetros incorretos")
    
    print("\n📋 Próximos passos:")
    print("1. Verifique se há arquivos em dados/cache/")
    print("2. Se não houver, tente mais tarde")
    print("3. Verifique https://nomads.ncep.noaa.gov/ manualmente")

if __name__ == "__main__":
    asyncio.run(main())