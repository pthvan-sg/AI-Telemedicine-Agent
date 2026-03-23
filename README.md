# MedScribe AI — Flask API

## Overview
MedScribe AI is a production-deployed NLP agent that analyzes 
doctor-patient transcripts and generates structured clinical summaries. 
Built with Flask, containerized via Docker, and orchestrated on Kubernetes. 
Integrates the OpenAI API for medical text understanding.

**Stack:** Python · Flask · Docker · Kubernetes · OpenAI API  
**Status:** Deployed and live

---

## Project Structure

```
ProjectHub/
├── app.py               ← Flask API (main app)
├── medical_summarizer.py← original CLI script (kept for reference)
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← container definition
├── .gitignore           ← keeps .env out of Git
├── .env                 ← YOUR secret API key (never share/commit)
└── README.md
```

---

## Step 1 — Create your .env file

Create a `.env` file in the project folder:
```
OPENAI_API_KEY=sk-your-key-here
```

---

## Step 2 — Build the Docker container

```powershell
docker build -t medscribe-agent .
```

---

## Step 3 — Run locally to test

```powershell
docker run --env-file .env -p 5000:5000 medscribe-agent
```

Test it:

```powershell
# Health check
curl http://localhost:5000/health

# Analyze sample transcript
curl -X POST http://localhost:5000/analyze/sample

# Analyze your own transcript
curl -X POST http://localhost:5000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"transcript\": \"Doctor: How are you?\nPatient: I have a headache.\"}"
```

---

## Step 4 — Push to Docker Hub

```powershell
# Login
docker login

# Tag with your Docker Hub username
docker tag medscribe-agent YOUR_DOCKERHUB_USERNAME/medscribe-agent:latest

# Push
docker push YOUR_DOCKERHUB_USERNAME/medscribe-agent:latest
```

---

## Step 5 — Deploy to Kubernetes

### Store your API key as a Kubernetes secret:
```powershell
kubectl create secret generic medscribe-secret `
  --from-literal=OPENAI_API_KEY=sk-your-key-here
```

### Apply the deployment:
```powershell
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
```

---

## API Endpoints

| Method | Endpoint          | Description                     |
|--------|-------------------|---------------------------------|
| GET    | /                 | API info                        |
| GET    | /health           | Health check (Kubernetes probe) |
| POST   | /analyze          | Analyze transcript (JSON body)  |
| POST   | /analyze/sample   | Analyze built-in sample         |

---

> AI-generated summaries are for documentation assistance only. Always verify with the treating physician.
