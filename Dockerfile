FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MEDSCRIBE_DB=/data/medscribe.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py medical_summarizer.py ./
COPY medscribe ./medscribe
COPY data ./data
RUN mkdir -p /data

EXPOSE 5000

# Agent runs call the LLM several times, so the default 30s worker timeout is too short.
CMD ["gunicorn", "--workers=2", "--timeout=180", "--bind=0.0.0.0:5000", "app:app"]
