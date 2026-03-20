FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY rag_pipeline.py .
COPY chroma_db/ ./chroma_db/

# Copy embedding model from your Mac into the image
COPY embedding_model/ ./embedding_model/

# Copy Ollama model from your Mac into the image
COPY ollama_models/ /root/.ollama/

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]