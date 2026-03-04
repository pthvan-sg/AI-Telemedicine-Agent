# Use official lightweight Python image
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy and install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app
COPY app.py .

# Expose the port Flask runs on
EXPOSE 5000

# Run with gunicorn (production-grade server, not Flask dev server)
# 2 worker processes, binds to all interfaces on port 5000
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:5000", "app:app"]
