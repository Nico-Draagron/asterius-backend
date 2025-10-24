# =============================
# Dockerfile para produção - Asterius Backend
# =============================

# Etapa 1: Build (instala dependências em ambiente limpo)
FROM python:3.11-slim AS builder

# Instala dependências do sistema necessárias para pacotes científicos e NOMADS
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libhdf5-dev libnetcdf-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia apenas requirements.txt para instalar dependências
COPY requirements.txt ./requirements.txt

# Instala dependências Python em um diretório temporário (para multi-stage)
RUN pip install --user --no-cache-dir -r requirements.txt

# Etapa 2: Runtime (imagem final enxuta)
FROM python:3.11-slim

# Instala dependências do sistema necessárias para runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends libhdf5-dev libnetcdf-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia dependências Python da etapa builder
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin:$PATH"

# Copia todo o código do backend para a imagem
COPY . ./

# Cria diretórios necessários para operação
RUN mkdir -p NOMADS/dados/cache NOMADS/dados/logs backend/data/logs modelo

# Exponha a porta 10000 para o serviço FastAPI/Uvicorn
EXPOSE 10000

# Comando padrão para iniciar o backend em produção
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
