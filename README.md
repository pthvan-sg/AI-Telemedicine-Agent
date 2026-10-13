# MedScribe AI

An agent that turns a doctor-patient transcript into a visit note a clinician can check. Every claim in the note has to be backed by a quote from the transcript, and drug safety checks run on the final note in code.

Stack: Python, Flask, OpenAI API (tool calling), MCP, SQLite, Docker, Kubernetes.

## What it does

The first version was one LLM call that returned JSON. It is still here as the baseline (`/analyze/baseline`). The current version is an agent loop:

1. Fetches stored patient history (long-term memory, SQLite).
2. Reads the transcript. Long transcripts are condensed with a running summary, and `search_transcript` recovers exact wording.
3. Calls tools: drug interactions, allergy conflicts, guideline retrieval (RAG), ICD-10 lookup, documentation gap check.
4. Submits a note with a verbatim evidence quote per item.
5. A verifier checks the note. Fabricated quotes, invented drugs or doses, patient speech used as a prescription, and made-up ICD codes are rejected, and the model must revise. Anything still unsupported after the revision budget is removed.
6. Safety alerts and documentation gaps are recomputed in code on the final note, so they do not depend on the model calling a tool.

Tools are served by an MCP server (`python -m medscribe.mcp_server`). Set `MEDSCRIBE_TOOL_BACKEND=mcp` to have the agent use it.

## Layout

```
app.py                  Flask API
medscribe/agent.py      agent loop
medscribe/verifier.py   grounding checks
medscribe/tools.py      tool registry
medscribe/rag.py        BM25 retrieval (swap for embeddings in this file only)
medscribe/memory.py     patient store
medscribe/context.py    chunking, running summary, transcript search
medscribe/mcp_server.py, mcp_backend.py
medscribe/baseline.py   original single-call version
data/                   demo drug KB, ICD-10 subset, guidelines, synthetic patients
```

## Run

```
cp .env.example .env        # add your key
pip install -r requirements.txt
python -m medscribe                               # sample transcript
python -m medscribe visit.txt --patient P-1001 --trace
python -m medscribe --baseline                    # baseline behaviour
docker build -t medscribe-agent . && docker run --env-file .env -p 5000:5000 medscribe-agent
curl -X POST localhost:5000/analyze/sample
```

API: `POST /analyze` with `{"transcript": "...", "patient_id": "P-1001", "save": false, "include_trace": false}`. Also `/analyze/baseline`, `/analyze/sample`, `GET /patients/<id>`, `/health`, `/ready`. Set `MEDSCRIBE_API_KEY` to require an `X-API-Key` header.

Kubernetes: create the secret as before, edit the image name in `deployment.yaml`, then `kubectl apply -f deployment.yaml`.

## Limits

The drug KB and ICD-10 list are small demo sets, not clinical resources. Patient data is synthetic, never put real PHI in this repo. Output is a draft for clinician review.
