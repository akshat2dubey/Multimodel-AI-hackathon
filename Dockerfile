FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JALNETRA_HOST=0.0.0.0 \
    JALNETRA_COOKIE_SECURE=1 \
    JALNETRA_AUTH=on \
    JALNETRA_STORAGE_REQUIRED=1

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["sh", "-c", "waitress-serve --listen=0.0.0.0:${PORT:-8080} dashboard.wsgi:app"]
