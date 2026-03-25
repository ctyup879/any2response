FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8765
ENV REQUEST_LOG_PATH=/app/last_request.json

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8765

CMD python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8765}
