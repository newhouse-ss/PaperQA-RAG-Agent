# Citation-grounded RAG Agent with Evaluation

> 日本語版のプロジェクト説明 / README.md はこちら： [README_日本語.md](./README_日本語.md)

## Project Overview

This is a cloud-native, adaptive RAG agent system that I developed end-to-end on
my own, from requirements definition through to deployment and operation.

Agent architecture: Built on LangGraph, it features autonomous query routing,
multi-hop reasoning, and self-correction of retrieval results, producing answers
with traceable, evidence-grounded citations.

AWS data platform: I built a robust data infrastructure spanning the entire flow
-- from ingestion into an S3 data lake, through an automated ETL pipeline on AWS
Glue (parsing, chunking, and dimensionality reduction of embedding vectors), to
persistence in Amazon RDS PostgreSQL with pgvector (HNSW index).

## Technical Stack Overview

An adaptive Retrieval-Augmented Generation system built on LangGraph that
answers multi-hop questions with traceable source citations.  The project
implements a complete cloud-native data pipeline: raw documents land in Amazon
S3, an AWS Glue ETL job parses, chunks, and embeds them into Amazon RDS
PostgreSQL with pgvector, and a FastAPI service with semantic caching serves
answers through an interactive Streamlit dashboard.  A companion evaluation
module generates synthetic test sets via knowledge-graph node tracking and
measures chain-coverage and step-coverage metrics.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [ETL Pipeline](#etl-pipeline)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the ETL Locally](#running-the-etl-locally)
  - [Running the ETL on AWS Glue](#running-the-etl-on-aws-glue)
  - [FastAPI Service](#fastapi-service)
  - [Streamlit Dashboard](#streamlit-dashboard)
- [Data Model](#data-model)
- [Vector Store Backends](#vector-store-backends)
- [Semantic Caching](#semantic-caching)
- [Evaluation Framework](#evaluation-framework)
- [AWS Setup Scripts](#aws-setup-scripts)
- [AWS Deployment (CDK)](#aws-deployment-cdk)
- [API Reference](#api-reference)

---

## Architecture Overview

![Architecture Overview](assets/archi_overview.png)

1. **Extract** -- Raw documents (PDF, HTML, plain text) are uploaded to an S3
   bucket serving as the data lake.
2. **Transform** -- An AWS Glue Python Shell job downloads each file, parses
   it, splits the text into overlapping character-based chunks, and calls the
   Google Generative AI embedding API (`gemini-embedding-001`) to produce
   768-dimensional vectors (reduced from 3072 via `output_dimensionality`).
3. **Load** -- Chunks with their embeddings and JSONB metadata are upserted
   into an RDS PostgreSQL table with an HNSW index for approximate
   nearest-neighbour search via pgvector.
4. **Serve** -- A FastAPI service backed by LangGraph performs adaptive
   retrieval, relevance grading, query rewriting, and citation-grounded
   answer generation.  An embedding-based semantic cache intercepts
   near-duplicate queries to cut latency.
5. **Dashboard** -- A Streamlit app provides a chat interface with expandable
   citation panels and cache management controls.

---

## Project Structure

```
.
├── rag_agent/                  # Core RAG engine (Python package)
│   ├── api.py                  #   FastAPI service with semantic cache
│   ├── cache.py                #   Embedding-based semantic cache
│   ├── config.py               #   Model names, chunking params, URL loader
│   ├── graph_builder.py        #   LangGraph state-machine construction
│   ├── loader.py               #   Document loader (PDF, HTML, local files)
│   ├── models.py               #   LLM and embedding model initialisation
│   ├── nodes.py                #   Node functions (router, grader, rewriter, generator)
│   ├── tools.py                #   Retriever tool with citation formatting
│   └── vectorstore.py          #   Vector store factory (pgvector / in-memory)
│
├── etl/                        # ETL pipeline
│   ├── glue_etl_job.py         #   AWS Glue Python Shell job script
│   ├── local_runner.py         #   Local testing entry point
│   └── schema.sql              #   PostgreSQL DDL (pgvector table + indexes)
│
├── evaluation/                 # Evaluation and synthetic data generation
│   ├── evaluation_schemas.py   #   Pydantic schemas (test cases, results)
│   ├── evaluation_metrics.py   #   Chain-coverage and step-coverage metrics
│   ├── evaluator.py            #   Orchestrator combining custom + Ragas metrics
│   └── kg_testset_generator.py #   KG-aware synthetic test-set generator
│
├── streamlit_app/              # Interactive dashboard
│   └── app.py                  #   Streamlit UI (chat, citations, cache controls)
│
├── infra/                      # AWS CDK infrastructure-as-code
│   ├── app.py                  #   CDK app entry point
│   ├── cdk.json                #   CDK configuration
│   ├── requirements.txt        #   CDK Python dependencies
│   └── stacks/
│       └── rag_service_stack.py#   S3, Glue, RDS/pgvector, ECS Fargate, ALB
│
├── scripts/                       # AWS provisioning and helper scripts
│   ├── aws_setup.py               #   Provision S3, RDS, Glue via boto3
│   ├── s3_upload.py               #   Upload local files to S3 data lake
│   ├── run_glue_job.py            #   Trigger and monitor Glue ETL job
│   └── check_rds.py               #   Verify RDS data (pgvector, row counts)
│
├── sample_docs/                   # Sample documents for testing
│
├── Dockerfile                  #   Container image for the FastAPI service
├── requirements.txt            #   Python dependencies
├── urls.txt                    #   Document source URLs (development mode)
└── README.md
```

---

## ETL Pipeline

The ETL pipeline lives in `etl/` and is designed to run as an **AWS Glue
Python Shell** job.  It is intentionally self-contained (no imports from the
`rag_agent` package) so it can execute inside the Glue runtime without
shipping the entire application.

### Pipeline Steps

| Stage | Component | Detail |
|---|---|---|
| Extract | `list_s3_objects` / `download_s3_bytes` | Lists and downloads files from the S3 data lake |
| Parse | `parse_pdf`, `parse_html`, `parse_text` | Converts raw bytes to structured text with metadata |
| Chunk | `chunk_text` / `split_documents` | Character-based overlapping chunking with stable hash IDs |
| Embed | `generate_embeddings` | Batched calls to Google `gemini-embedding-001` (768-dim via `output_dimensionality`) |
| Load | `ensure_schema` / `upsert_chunks` | Upserts into PostgreSQL; skips duplicates by `chunk_id` |

### Glue Job Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--S3_BUCKET` | Yes | -- | Source S3 bucket |
| `--S3_PREFIX` | No | `raw/` | Key prefix to scan |
| `--DB_HOST` | Yes* | -- | RDS endpoint |
| `--DB_NAME` | No | `ragdb` | Database name |
| `--DB_USER` | Yes* | -- | Database user |
| `--DB_PASSWORD` | Yes* | -- | Database password |
| `--DB_SECRET_ARN` | Yes* | -- | Secrets Manager ARN (overrides user/password) |
| `--GOOGLE_API_KEY` | Yes | -- | Google AI Studio API key |
| `--CHUNK_SIZE` | No | `1024` | Characters per chunk |
| `--CHUNK_OVERLAP` | No | `50` | Overlap characters |

*Provide either `DB_SECRET_ARN` or the `DB_HOST` / `DB_USER` / `DB_PASSWORD`
triple.

### Glue Additional Python Modules

```
pypdf,beautifulsoup4,google-generativeai,pg8000
```

All modules are pure Python to ensure compatibility with the Glue Python Shell
runtime (no C compilation available).  `pg8000` replaces `psycopg2-binary` and
connects to RDS over SSL.

---

## Key Features

- **Cloud-native ETL** -- S3 data lake, AWS Glue transform, RDS pgvector
  warehouse; a standard Extract-Transform-Load pipeline with full
  infrastructure-as-code.
- **Adaptive routing** -- the LLM autonomously decides between direct
  generation and retrieval-augmented generation based on query intent.
- **Self-correcting retrieval** -- a grading step evaluates retrieved chunks
  and triggers query rewriting when context quality is low.
- **Citation grounding** -- every answer embeds structured source references
  (URL, page, chunk ID, text snippet) extracted deterministically from
  retriever output.
- **Semantic caching** -- an embedding-similarity cache returns previously
  computed answers for near-duplicate queries, reducing latency and cost.
- **Dual vector-store backend** -- pgvector for production (populated by
  Glue ETL), in-memory for development (loads from `urls.txt` on startup).
- **KG-aware evaluation** -- synthetic test-set generation with
  knowledge-graph node tracking; chain-coverage and step-coverage metrics.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- A Google AI Studio API key
  ([get one here](https://aistudio.google.com/apikey))
- PostgreSQL 16 with the `pgvector` extension (for ETL and production use)
- AWS CLI and CDK CLI (for cloud deployment only)

### Installation

```bash
git clone https://github.com/<your-username>/doc-Q-A-RAG-Agent.git
cd doc-Q-A-RAG-Agent

# virtual environment
conda create -n ragent python=3.10 && conda activate ragent
# or: python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# for ETL local runner (optional)
pip install google-generativeai psycopg2-binary
```

### Configuration

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key |
| `PGVECTOR_CONNECTION_STRING` | No | PostgreSQL URI; omit to use in-memory store |
| `PGVECTOR_COLLECTION` | No | pgvector table name (default: `documents`) |

```bash
export GOOGLE_API_KEY="your_key"

# production
export PGVECTOR_CONNECTION_STRING="postgresql://user:pass@host:5432/ragdb"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your_key"
```

---

## Usage

### Running the ETL Locally

Create a directory with sample documents and run the local runner:

```bash
mkdir sample_docs
# place PDF / HTML / TXT files into sample_docs/

# dry run (parse + chunk only, no embedding or DB)
python -m etl.local_runner --input_dir ./sample_docs --dry_run

# full run (requires PostgreSQL and GOOGLE_API_KEY)
python -m etl.local_runner \
    --input_dir ./sample_docs \
    --db_host localhost \
    --db_name ragdb \
    --db_user postgres \
    --db_password secret \
    --google_api_key $GOOGLE_API_KEY
```

### Running the ETL on AWS Glue

1. Upload raw documents to your S3 bucket under the `raw/` prefix.
2. Upload `etl/glue_etl_job.py` to `s3://<bucket>/glue-scripts/`.
3. Create a Glue Python Shell job pointing to the script with the parameters
   listed above.
4. Run the job manually or attach an EventBridge trigger.

Or deploy via CDK (see [AWS Deployment](#aws-deployment-cdk)) which provisions
the job automatically.

### FastAPI Service

```bash
uvicorn rag_agent.api:app --host 0.0.0.0 --port 8000
```

### Streamlit Dashboard

Start the FastAPI service first, then:

```bash
streamlit run streamlit_app/app.py
```

---

## Data Model

The ETL pipeline writes into the following PostgreSQL table:

```sql
CREATE TABLE documents (
    id          SERIAL       PRIMARY KEY,
    chunk_id    TEXT         UNIQUE NOT NULL,
    content     TEXT         NOT NULL,
    metadata    JSONB        DEFAULT '{}'::jsonb,
    embedding   vector(768),
    created_at  TIMESTAMPTZ  DEFAULT now()
);
```

The `metadata` column stores:

```json
{
  "source": "s3://bucket/raw/report.pdf",
  "page": 3,
  "title": "Annual Report",
  "chunk_index": 0
}
```

Indexes:
- **HNSW** on `embedding` for approximate cosine-similarity search.
- **B-tree** on `metadata->>'source'` for deduplication and source filtering.

Full DDL is in `etl/schema.sql`.

---

## Vector Store Backends

| Backend | When | Data Source | Persistence |
|---|---|---|---|
| **pgvector** | `PGVECTOR_CONNECTION_STRING` is set | Glue ETL writes; API reads | PostgreSQL |
| **InMemory** | variable not set | Loads from `urls.txt` on startup | None |

---

## Semantic Caching

The FastAPI service wraps the LangGraph agent with an embedding-based cache
(`rag_agent/cache.py`).

1. Incoming query is embedded and compared (cosine similarity) against cached
   entries.
2. If similarity exceeds the threshold (default 0.92) the cached answer is
   returned immediately.
3. Otherwise the full agent pipeline runs and the result is stored.

Endpoints for runtime management:
- `GET  /v1/cache/stats` -- current entry count
- `DELETE /v1/cache` -- flush

---

## Evaluation Framework

The `evaluation/` package measures multi-hop retrieval quality:

| Metric | Definition |
|---|---|
| Chain coverage | Fraction of gold-standard chunks retrieved |
| Step coverage | Fraction of reasoning steps with at least one required chunk retrieved |
| Answer correctness | Ragas semantic similarity (system answer vs gold answer) |
| Faithfulness | Ragas grounding score |
| Context precision / recall | Ragas context-level metrics |

`kg_testset_generator.py` extends Ragas to track which knowledge-graph nodes
support each generated question-answer pair, enabling automatic computation
of chain and step coverage on synthetic data.

---

## AWS Setup Scripts

The `scripts/` directory provides boto3-based helpers for quick provisioning
without CDK:

| Script | Purpose |
|---|---|
| `aws_setup.py` | Creates S3 bucket, RDS PostgreSQL (pgvector), IAM role, and Glue job |
| `s3_upload.py` | Uploads a local directory to the S3 data lake under `raw/` |
| `run_glue_job.py` | Triggers the Glue ETL job and polls for completion |
| `check_rds.py` | Connects to RDS, verifies pgvector extension, table, and row counts |

```bash
# 1. Provision all AWS resources (~5 min for RDS)
python scripts/aws_setup.py

# 2. Upload documents
python scripts/s3_upload.py ./sample_docs

# 3. Run ETL
python scripts/run_glue_job.py --wait

# 4. Verify
python scripts/check_rds.py
```

---

## AWS Deployment (CDK)

The `infra/` directory contains an AWS CDK stack that provisions:

| Resource | Service | Detail |
|---|---|---|
| Network | VPC | 2 AZs, public + private subnets, NAT |
| Data lake | S3 | Bucket for raw documents |
| ETL | Glue Python Shell | Parses, chunks, embeds, loads |
| Database | RDS PostgreSQL 16 | pgvector extension, db.t4g.micro |
| API | ECS Fargate | FastAPI container behind ALB |
| Secrets | Secrets Manager | Google API key, DB credentials |

```bash
cd infra
pip install -r requirements.txt
cdk bootstrap   # first time only
cdk deploy
```

After deployment:
1. Upload documents to the S3 bucket under `raw/`.
2. Run the Glue job from the AWS console or CLI.
3. Access the API via the ALB DNS name output.

---

## Rate Limiting

The `/v1/chat` endpoint is rate-limited to **30 requests / minute per IP**
via [slowapi](https://github.com/laurentS/slowapi).  Exceeding the limit
returns HTTP 429:

```json
{"detail": "Rate limit exceeded: 30 per 1 minute"}
```

Implementation: `rag_agent/api.py` creates a `Limiter(key_func=get_remote_address)`
and decorates the chat endpoint with `@limiter.limit("30/minute")`.

---

## API Reference

### POST /v1/chat

Request:

```json
{
  "message": "How does Money Forward use AI?",
  "timeout_s": 60
}
```

Response:

```json
{
  "trace_id": "a1b2c3d4-...",
  "answer": "Money Forward uses AI to ...",
  "citations": [
    {
      "source": "s3://bucket/raw/report.pdf",
      "title": "AI Strategy",
      "page": 3,
      "chunk": "12",
      "snippet": "Money Forward leverages ..."
    }
  ],
  "cached": false
}
```

### GET /healthz

Returns `{"status": "ok"}`.

### GET /v1/cache/stats

Returns `{"entries": 42}`.

### DELETE /v1/cache

Flushes the semantic cache. Returns `{"status": "cleared"}`.

---