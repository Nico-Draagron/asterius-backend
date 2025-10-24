#!/usr/bin/env python3
"""
Script para executar o servidor Asterius
"""

if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    
    # Adiciona o diretório atual ao path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from main import app
    
    print("🚀 Iniciando Asterius Backend...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001, 
        reload=False,
        log_level="info"
    )