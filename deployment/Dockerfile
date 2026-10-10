FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic data + build the RAG index at image build time so the
# container is demoable immediately on first run.
RUN python data_generation/generate_synthetic_data.py && \
    python -m rag.ingest

EXPOSE 8501

# python:slim has no curl, so the health probe uses the Python stdlib instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
