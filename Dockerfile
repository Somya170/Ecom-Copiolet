FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000

WORKDIR /app

# Build tools aur SQLite utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code & assets copy
COPY app.py graph.py chainlit.md ./
COPY .chainlit ./.chainlit
COPY public ./public

# DB files image me default seed ke liye copy
COPY ecommerce_oltp.db ecommerce_analytics.duckdb ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000 || exit 1

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]