# MDB Agent Builder

A production-ready framework for building and deploying LLM agents with multiple agent types, multi-agent coordination, advanced memory systems, and governance controls. Built on LangChain, LangGraph, and MongoDB.

## Overview

MDB Agent Builder enables you to:

- **Build diverse agent types** — ReAct, Tool-Call, Reflection, Plan-Execute-Replan, and Long-Term Memory agents, all via YAML
- **Coordinate multiple agents** — configure handoffs between agents so they can route to each other based on conversation context
- **Scale across workers** — Gunicorn multi-worker support with MongoDB-backed conversation state and checkpointing
- **Add long-term memory** — episodic (verbatim) and observational (distilled) memory with MongoDB Atlas Vector Search
- **Enforce governance** — access policies, prompt injection detection, PII redaction, and audit logging
- **Integrate any LLM or tool** — pluggable adapters for Anthropic, Bedrock, Fireworks, Cohere, Grove, and 10+ other providers
- **Persist across restarts** — MongoDB checkpointing for durable graph and conversation state

---

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **MongoDB** — local (`mongodb://localhost:27017`) or Atlas cluster
- **An LLM provider** — one of: Anthropic, OpenAI, Bedrock, Fireworks, Cohere, Together, Azure, Ollama, SageMaker, or Grove

---

## Quick Start

The fastest way to get running:

```bash
# 1. Clone and enter the project
git clone https://github.com/mongodb/mdb-agent-builder.git
cd mdb-agent-builder

# 2. Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials (LLM API keys, MongoDB URI, etc.)
# Generate a Flask secret key:
#   python3 -c "import secrets; print(secrets.token_hex(32))"

# 4. Create default config and directories
make create-config

# 5. Run the dev server (Flask, single process)
make run
# ...or directly:
# agent-builder serve --config config/agents.yaml

# 6. Test the health endpoint
curl http://localhost:5000/health
```

Visit `http://localhost:5000/health` and start chatting at `POST /chat`.

> **Dev vs. production:** `agent-builder serve` (and `make run`) start the single-process Flask development server, bound to `127.0.0.1` by default (pass `--host 0.0.0.0` to expose it on the network). For multi-worker deployments use Gunicorn via `make serve-prod`, the Docker image, or a direct `gunicorn agent_builder.wsgi:application` invocation. See [Multi-Worker Deployments](#multi-worker-deployments) below.

---

## Full Setup Guide

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/mongodb/mdb-agent-builder.git
cd mdb-agent-builder
```

**Option A: Using the Makefile (recommended)**

```bash
# Create virtual environment
make setup-env

# Activate the environment
source .venv/bin/activate

# Install with dev dependencies (includes linters, formatters, test tools)
make install
```

**Option B: Manual**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### Step 2: Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your settings. At minimum you need:

```bash
# MongoDB (required)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_builder

# At least one LLM provider key (choose the one you use)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
FIREWORKS_API_KEY=...
COHERE_API_KEY=...
TOGETHER_API_KEY=...

# Flask secret key — generate one:
#   python3 -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=<your-generated-key>

# Admin token — required for admin endpoints (GET /threads, global POST /reset).
# Leave unset to disable admin operations entirely. Generate like the secret key.
MDB_ADMIN_TOKEN=<your-generated-token>

# Server settings
LOG_LEVEL=INFO
PORT=5000
```

For a Grove API gateway setup, also set:

```bash
GROVE_API_BASE=https://grove.example.com/v1
GROVE_API_KEY=...
```

### Step 3: Create configuration

```bash
make create-config
```

This creates the `config/`, `logs/`, and `prompts/` directories with a default `config/agents.yaml`. Edit `config/agents.yaml` to configure your LLM, tools, and agent type. See [Configuration](#configuration) for details.

**Validate your config:**

```bash
make validate-config
```

### Step 4: Run the application

**Development (single process, Flask dev server):**

```bash
make run
# or: agent-builder serve --config config/agents.yaml --port 5000
```

The server starts at `http://localhost:5000`.

**Production (multi-worker, Gunicorn):**

```bash
make serve-prod GUNICORN_WORKERS=8 PORT=5000
# or directly:
# AGENT_CONFIG_PATH=config/agents.yaml gunicorn --workers 8 --bind 0.0.0.0:5000 agent_builder.wsgi:application
```

For multi-worker deployments, configure `state:` or `governance:` in your YAML to enable cross-worker session sharing via MongoDB. See [Multi-Worker Deployments](#multi-worker-deployments).

### Step 5: Send a chat request

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -H "X-Requested-With: XMLHttpRequest" \
  -d '{
    "message": "Hello, what can you help me with?",
    "config": {
      "thread_id": "my-first-conversation",
      "identity": {
        "tenant_id": "default",
        "user_id": "dev-user"
      }
    }
  }'
```

> **Note:** The `X-Requested-With: XMLHttpRequest` header is required for state-changing endpoints (like `/reset`) as a CSRF protection measure. It is recommended on all POST requests.

---

## Docker

The Docker image runs Gunicorn (multi-worker) via `startup.sh`, ships with example configs, and exposes a `/health` healthcheck.

### Building and running

```bash
# Build the image (defaults to Docker; use CONTAINER_RUNTIME=podman for Podman)
make docker-build
make docker-build CONTAINER_RUNTIME=podman   # if you use Podman

# Run the container
make docker-run PORT=5000 GUNICORN_WORKERS=8
make docker-run CONTAINER_RUNTIME=podman GUNICORN_WORKERS=8

# Debug mode (interactive shell)
make docker-debug
```

`make docker-run` automatically loads your `.env` file if present and mounts your local `config/`, `logs/`, and `prompts/` directories as volumes.

### Direct commands

```bash
# Build
docker build -t mdb-agent-builder .

# Run
docker run -p 5000:5000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/prompts:/app/prompts \
  --env-file .env \
  -e AGENT_CONFIG_PATH=/app/config/agents.yaml \
  -e GUNICORN_WORKERS=8 \
  mdb-agent-builder
```

The `CONTAINER_RUNTIME` Makefile variable also applies to `docker-run` and `docker-debug`. Override it with `make docker-run CONTAINER_RUNTIME=podman` to use Podman instead of Docker across all three targets.

---

## Using the Makefile

Common workflows are wrapped as `make` targets. Run `make help` for the full list.

| Target | Description |
|--------|-------------|
| `make setup-env` | Create a virtual environment in `.venv` |
| `make install` | Install the package with dev extras (run inside activated venv) |
| `make create-config` | Scaffold `config/`, `logs/`, `prompts/` with default `agents.yaml` |
| `make validate-config` | Validate the YAML configuration |
| `make create-agent` | Interactive wizard to add a new agent to `agents.yaml` |
| `make add-tool` | Interactive wizard to add a new tool to `agents.yaml` |
| `make run` / `make serve` | Run Flask dev server (single process) |
| `make serve-prod` | Run Gunicorn multi-worker (production-like) |
| `make docker-build` | Build the container image |
| `make docker-run` | Run the container |
| `make docker-debug` | Start container in interactive shell mode |
| `make test` | Run the test suite |
| `make lint` / `make format` | Lint / format the code |
| `make clean` | Remove build artifacts and caches |
| `make verify` | Verify installation and configuration |

### Overridable variables

All targets that use these variables accept overrides on the command line:

```bash
# Container runtime (docker or podman) — applies to docker-build, docker-run, docker-debug
make docker-build CONTAINER_RUNTIME=podman

# Server settings — applies to serve-prod, docker-run, docker-debug
make serve-prod GUNICORN_WORKERS=8 PORT=8080 GUNICORN_TIMEOUT=180
make docker-run GUNICORN_WORKERS=8 PORT=8080

# Configuration path — applies to validate-config
make validate-config CONFIG_PATH=examples/multi_agent_customer_support.yaml
```

| Variable | Default | Applies to |
|----------|---------|------------|
| `CONTAINER_RUNTIME` | `docker` | `docker-build`, `docker-run`, `docker-debug` |
| `PORT` | `5000` | `run`, `serve`, `serve-prod`, `docker-run`, `docker-debug` |
| `GUNICORN_WORKERS` | `4` | `serve-prod`, `docker-run`, `docker-debug` |
| `GUNICORN_TIMEOUT` | `120` | `serve-prod`, `docker-run`, `docker-debug` |
| `CONFIG_PATH` | `config/agents.yaml` | `validate-config` (override via `make validate-config CONFIG_PATH=...`) |

---

## Configuration

### YAML Structure

Agent configs are YAML files with these top-level sections:

```yaml
embeddings:      # Embedding models (for vector search, memory)
  - name: voyage
    provider: voyageai
    model_name: voyage-3.5-lite

llms:            # Language models
  - name: claude-3.5
    provider: bedrock
    model_name: anthropic.claude-3-5-sonnet-20240620-v1:0
    temperature: 0.7
    max_tokens: 4096

tools:           # Tools available to agents
  - name: search
    tool_type: vector_search
    namespace: my_db.products
    connection_str: ${MONGODB_URI}
    embedding_model: voyage

memory:          # Optional: long-term memory adapters
  - name: recall
    memory_type: episodic
    namespace: my_db.episodes
    embedding_model: voyage

agent:           # Single-agent mode
  name: my_agent
  agent_type: react
  llm: claude-3.5
  tools: [search]
  system_prompt_path: ./prompts/rag_system_prompt.txt

# OR for multi-agent mode:
agents:          # Multi-agent mode (mutually exclusive with agent:)
  - name: triage
    agent_type: react
    llm: claude-3.5
    handoffs:
      - name: billing
        description: "Route billing questions here"

checkpointer:    # MongoDB checkpointing for durable state
  connection_str: ${MONGODB_URI}
  db_name: agent_state
  collection_name: checkpoints

state:           # Standalone session state for multi-worker deployments
  enabled: true
  connection_str: ${MONGODB_URI}

governance:      # Optional: policies, audit, PII redaction
  enabled: true
  connection_str: ${MONGODB_URI}
  default_policy:
    permissions: ["*"]
  policy:
    provider: mongodb
  audit:
    enabled: true
  state:
    enabled: true
```

> **Note:** Environment variable names resolved from YAML configs are restricted to an allowlist for security. Variables matching patterns like `MONGODB_.*`, `OPENAI_.*`, `ANTHROPIC_.*`, `GROVE_.*`, `OLLAMA_.*`, etc. are permitted. Customize the list by setting the `YAML_ENV_VAR_ALLOWLIST` environment variable to a comma-separated list of regex patterns.

### Environment Variables

```bash
# MongoDB (required)
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true
MONGODB_DATABASE=agent_builder

# LLM API keys (set the ones you need)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
FIREWORKS_API_KEY=...
COHERE_API_KEY=...
TOGETHER_API_KEY=...
VOYAGEAI_API_KEY=...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Grove API gateway (OpenAI-compatible LLM gateway)
GROVE_API_BASE=https://grove.example.com/v1
GROVE_API_KEY=...

# Flask / server settings
FLASK_SECRET_KEY=<generate with: python3 -c "import secrets; print(secrets.token_hex(32))">
FLASK_ENV=production           # Set in production to enforce FLASK_SECRET_KEY and block debug mode
MDB_ADMIN_TOKEN=<token>       # Required for GET /threads and global POST /reset; unset = admin ops disabled
AGENT_CONFIG_PATH=config/agents.yaml
LOG_LEVEL=INFO
PORT=5000

# Gunicorn (for serve-prod and Docker)
GUNICORN_WORKERS=4
GUNICORN_TIMEOUT=120

# Checkpointing (optional)
CHECKPOINT_ENABLED=false
CHECKPOINT_PROVIDER=mongodb
CHECKPOINT_URI=${MONGODB_URI}
CHECKPOINT_DB=${MONGODB_DATABASE}
```

### LLM Providers

Models are declared under `llms:` and referenced by name from agents.

| Provider | Key / Config | Notes |
|----------|-------------|-------|
| `bedrock` | AWS credentials in environment | Amazon Bedrock chat models |
| `anthropic` | `ANTHROPIC_API_KEY` | Anthropic Claude |
| `openai` | `OPENAI_API_KEY` | OpenAI GPT models |
| `fireworks` | `FIREWORKS_API_KEY` | Fireworks AI |
| `together` | `TOGETHER_API_KEY` | Together AI |
| `cohere` | `COHERE_API_KEY` | Cohere |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | Azure OpenAI |
| `ollama` | None (local) | Local Ollama models. Defaults to `http://localhost:11434` |
| `sagemaker` | `additional_kwargs.endpoint_name` | AWS SageMaker endpoints |
| `grove` | `GROVE_API_BASE`, `GROVE_API_KEY` | Grove API gateway (OpenAI-compatible) |

#### Example LLM configuration

```yaml
llms:
  - name: my-claude
    provider: anthropic
    model_name: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096

  - name: my-gpt
    provider: openai
    model_name: gpt-4o
    temperature: 0.3

  - name: local-llama
    provider: ollama
    model_name: llama3
    additional_kwargs:
      base_url: http://localhost:11434
```

#### Grove API Gateway

Grove is an OpenAI-compatible LLM gateway that fronts one or more upstream model providers behind a single endpoint.

```yaml
llms:
  - name: grove-claude
    provider: grove
    model_name: claude-3-5-sonnet        # model id as exposed by Grove
    temperature: 0.7
    max_tokens: 2048
    additional_kwargs:
      base_url: ${GROVE_API_BASE}        # e.g. https://grove.example.com/v1
      api_key: ${GROVE_API_KEY}
      default_headers:                   # optional extra gateway headers
        x-tenant-id: acme
```

Resolution order for `base_url`: `additional_kwargs.base_url` → `GROVE_API_BASE` → `GROVE_API_GATEWAY_URL`. Required. Resolution order for `api_key`: `additional_kwargs.api_key` → `GROVE_API_KEY` → a placeholder (for unauthenticated gateways). Any other keys under `additional_kwargs` (e.g. `default_headers`, `organization`, `timeout`) are passed through to the OpenAI-compatible client.

---

## Agent Types

### Available Types

| Value | Description |
|-------|-------------|
| `react` | ReAct agent — thinks step-by-step using tools in a reasoning loop |
| `tool_call` | Uses OpenAI-style tool calling (alias for `react`) |
| `reflect` | Generate-reflect loop that reviews and improves its own answers |
| `plan_execute_replan` | Creates a plan, executes steps, and replans as needed |
| `long_term_memory` | Agent with vector store-backed long-term memory |

### Single-Agent Configuration

```yaml
agent:
  name: my_agent
  agent_type: react              # Required
  llm: my-llm                    # Reference to an LLM defined above
  system_prompt: |
    You are a helpful assistant that can search product information.
  system_prompt_path: ./prompts/rag_system_prompt.txt  # Alternative to system_prompt
  tools: [search, calculator]    # Optional: list of tool names

  # For reflect agents
  reflection_prompt: |
    Review and improve your previous response for accuracy and clarity.
  reflection_prompt_path: ./prompts/reflection_prompt.txt
  
  # For long-term memory agents
  episodic_memory: recall        # Reference to a memory adapter
  observational_memory: analysis

  # For connecting to MongoDB directly (legacy long-term memory)
  connection_str: ${MONGODB_URI}
  namespace: my_db.memories
```

> **Security note:** Prompt files are loaded relative to the working directory with path traversal protection. Attempts to escape the working directory (e.g., `../../etc/passwd`) will be rejected.

### Multi-Agent Configuration

Use the `agents:` key (plural) to define multiple agents that can hand off to each other. Mutually exclusive with the singular `agent:` key.

```yaml
agents:
  - name: triage_agent
    agent_type: react
    llm: claude-3.5
    system_prompt: "Classify requests and route to the right specialist."
    handoffs:
      - name: billing_agent
        description: "Transfer for billing or payment questions"
      - name: technical_agent
        description: "Transfer for technical support"

  - name: billing_agent
    agent_type: react
    llm: claude-3.5
    system_prompt: "Handle billing and payment questions."
    tools: [billing_search]
    handoffs:
      - name: triage_agent
        description: "Return to triage for non-billing questions"

  - name: technical_agent
    agent_type: react
    llm: claude-3.5
    system_prompt: "Handle technical support issues."
    tools: [product_knowledge]

entry_agent: triage_agent    # Which agent receives the first message (defaults to first in list)
```

When an agent calls a handoff tool (e.g., `transfer_to_billing_agent`), execution routes to the target agent in the same conversation thread. See `examples/multi_agent_customer_support.yaml` for a complete working example.

---

## Multi-Worker Deployments

### Process-Local History (Single Worker)

By default, conversation history is stored in-memory within each process. This works for single-worker deployments but does not survive restarts or scale across workers.

### Cross-Worker Session State

For Gunicorn multi-worker deployments, enable either a standalone `state:` config or the full `governance:` config to share conversation history across workers via MongoDB:

```yaml
# Option 1: Standalone state (lightweight, no governance)
state:
  enabled: true
  connection_str: ${MONGODB_URI}
  db_name: agent_state
  collection_name: agent_sessions

# Option 2: Full governance (policies, audit, state all enabled)
governance:
  enabled: true
  connection_str: ${MONGODB_URI}
  state:
    enabled: true
  audit:
    enabled: true
```

When governance is enabled alongside standalone state, governance takes precedence for the state provider.

### Starting Multiple Workers

```bash
# Via the Makefile (recommended)
make serve-prod GUNICORN_WORKERS=8 GUNICORN_TIMEOUT=120 PORT=5000

# Gunicorn directly (what the Docker image's startup.sh uses)
AGENT_CONFIG_PATH=config/agents.yaml \
  gunicorn --workers 8 --timeout 120 --bind 0.0.0.0:5000 \
  agent_builder.wsgi:application

# Via Docker
make docker-run GUNICORN_WORKERS=8 PORT=5000
```

The `/health` endpoint responds over HTTP without cookies or session state, so load balancers can route traffic cleanly. Each worker reads and writes chat history to the same MongoDB collection, keeping conversation state consistent across the fleet.

---

## Memory Systems

### Episodic Memory

Stores verbatim conversation snippets (what was said, when, how the user felt). Retrieved by semantic similarity.

```yaml
memory:
  - name: episode_recall
    memory_type: episodic
    connection_str: ${MONGODB_URI}
    namespace: my_db.episodes
    embedding_model: voyage          # Reference to an embedding model
    index_name: episodic_index

agent:
  agent_type: long_term_memory
  llm: claude-3.5
  episodic_memory: episode_recall    # Reference to the memory adapter
```

### Observational Memory

Uses an LLM to distil raw conversation into structured facts (user preferences, behavioral patterns, goals). Each observation is stored as a separate vector-search document for granular retrieval.

```yaml
memory:
  - name: observations
    memory_type: observational
    connection_str: ${MONGODB_URI}
    namespace: my_db.observations
    embedding_model: voyage
    llm: claude-3.5                # LLM used to extract observations
    index_name: observational_index
    extraction_prompt: |           # Optional — overrides the default prompt
      Extract key user facts from this conversation: {text}

agent:
  agent_type: long_term_memory
  llm: claude-3.5
  observational_memory: observations
  episodic_memory: episode_recall   # Can use both together
```

---

## API Endpoints

All endpoints expect `Content-Type: application/json`. State-changing endpoints (`/reset`) require the `X-Requested-With: XMLHttpRequest` header as CSRF protection. Administrative operations (`GET /threads`, global `POST /reset`) additionally require the `X-Admin-Token` header to match the `MDB_ADMIN_TOKEN` environment variable — when that variable is unset, they are disabled.

The application enforces a 1 MB request body limit and rate limits `/chat` (60 requests/minute) and `/reset` (30 requests/minute). Rate-limit buckets are keyed on the source IP address plus the asserted tenant and user, so rotating identities does not bypass the limit.

### Health Check

```bash
curl http://localhost:5000/health

# Response (200):
# {"status": "healthy", "agent_loaded": true}
# Response (503 if agent not loaded):
# {"status": "unhealthy", "agent_loaded": false}
```

### Chat

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "config": {
      "thread_id": "user-123-conv-1",
      "identity": {
        "tenant_id": "acme-corp",
        "user_id": "user@example.com",
        "roles": ["customer"]
      }
    }
  }'
```

**Response (200):**

```json
{
  "response": "Paris is the capital of France.",
  "history": [
    ["user", "What is the capital of France?"],
    ["assistant", "Paris is the capital of France."]
  ],
  "thread_id": "user-123-conv-1"
}
```

**Error responses:**

```json
// 400 — missing message field
{"error": "Missing required field: message"}

// 403 — blocked by governance guardrail
{"error": "Request blocked by input guardrail", "reason": "Blocked topic detected: ...", "thread_id": "..."}

// 404 — thread not found or cross-tenant access denied
{"error": "Thread not found", "thread_id": "..."}

// 415 — wrong content type
{"error": "Content-Type must be application/json"}

// 429 — rate limit exceeded
{"error": "Too many requests — rate limit exceeded"}

// 500 — agent invocation failed (no internal details exposed)
{"error": "Agent invocation failed", "thread_id": "..."}
```

### Reset Conversation

```bash
# Reset a specific thread
curl -X POST http://localhost:5000/reset \
  -H "Content-Type: application/json" \
  -H "X-Requested-With: XMLHttpRequest" \
  -d '{"thread_id": "user-123-conv-1"}'

# Response:
# {"status": "success", "message": "Chat history reset for thread user-123-conv-1"}
```

Thread ownership is verified against the state provider when available, and the durable copy of the history is cleared as well as the in-process copy.

Omitting `thread_id` resets **all** threads — this is an administrative operation that additionally requires the `X-Admin-Token` header to match the `MDB_ADMIN_TOKEN` environment variable:

```bash
curl -X POST http://localhost:5000/reset \
  -H "Content-Type: application/json" \
  -H "X-Requested-With: XMLHttpRequest" \
  -H "X-Admin-Token: $MDB_ADMIN_TOKEN" \
  -d '{}'
```

If `MDB_ADMIN_TOKEN` is not set, global reset (and the `/threads` endpoint below) are disabled entirely.

### List Active Threads (admin)

Thread IDs grant access to conversation history, so listing them requires the admin token:

```bash
curl http://localhost:5000/threads -H "X-Admin-Token: $MDB_ADMIN_TOKEN"

# Response:
# {"status": "success", "threads": ["thread-1", "thread-2"], "count": 2}
```

> **Note:** Thread listing shows in-process threads only. When using a MongoDB state provider, threads persisted across workers are not reflected in this endpoint's response (it lists process-local threads).

---

## Governance & Security

### Access Policies

Define role-based or tenant-based policies in MongoDB:

```bash
# Insert via mongosh
db.agent_policies.insertOne({
  tenant_id: "acme-corp",
  role: "support",
  permissions: ["tools.call.search", "tools.call.send_email"],
  denied_tools: ["delete_user"],
  blocked_topics: ["salary", "passwords"],
  retrieval_filters: { classification: "public" },
  pii_redaction: true,
  prompt_injection_detection: true
})
```

Enable governance in your YAML config:

```yaml
governance:
  enabled: true
  connection_str: ${MONGODB_URI}
  db_name: agent_control_plane
  default_policy:
    permissions: ["*"]
  policy:
    provider: mongodb
    collection_name: agent_policies
```

### Guardrails

Automatically applied when `governance.enabled: true`:

- **Blocked Topics** — reject requests mentioning restricted words (case-insensitive)
- **Prompt Injection Detection** — pattern-match common jailbreak attempts
- **PII Redaction** — redact emails and phone numbers from input and output
- **Tool Access Control** — only call tools the user's policy permits; denied tools are blocked regardless of wildcard permissions
- **Tenant Isolation** — all data retrieval is scoped by `tenant_id` and `user_id`

### Security Headers

All API responses include the following security headers:

| Header | Value |
|--------|-------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `0` |
| `Cache-Control` | `no-store` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `X-Permitted-Cross-Domain-Policies` | `none` |
| `Content-Security-Policy` | `default-src 'none'; frame-ancestors 'none'` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` (HTTPS responses only) |

### Server Hardening Defaults

- **Localhost bind by default** — `agent-builder serve` and the Flask dev server bind `127.0.0.1`; exposing on the network requires an explicit `--host 0.0.0.0`. Gunicorn/Docker deployments bind `0.0.0.0` explicitly in `startup.sh`.
- **Debug mode blocked in production** — `AgentApp.run(debug=True)` raises when `FLASK_ENV=production` (the Werkzeug debugger allows arbitrary code execution from the browser).
- **Admin operations are opt-in** — `GET /threads` and global `POST /reset` require `X-Admin-Token` matching `MDB_ADMIN_TOKEN` (constant-time comparison). With no token configured, they are disabled — there is no default credential.
- **No message content in logs** — `/chat` logs message length and thread ID only, so PII never reaches the logs even when governance redaction is off.
- **YAML env-var allowlist** — `${VAR}` references in YAML configs resolve only variables matching `YAML_ENV_VAR_ALLOWLIST` patterns, so a tampered config cannot exfiltrate arbitrary server secrets.
- **MDB prefix for admin token** — environment variable is now `MDB_ADMIN_TOKEN` instead of the previous `MAAP_ADMIN_TOKEN`.
- **Sanitized connection strings** — MongoDB URIs are credential-masked before logging.

> **Deployment note:** the API has no built-in end-user authentication — `tenant_id`/`user_id` are asserted by the client. Deploy behind a trusted gateway or reverse proxy that authenticates callers and injects identity; do not expose the service directly to untrusted clients. If you use the `nl_to_mql` tool, scope the MongoDB user it connects with to read-only access on the intended database.

### Audit Logging

All governance events are logged to MongoDB (`agent_audit_events` by default):

```json
{
  "event_type": "guardrail.input",
  "tenant_id": "acme-corp",
  "user_id": "user@example.com",
  "agent_id": "triage_agent",
  "thread_id": "conv-123",
  "payload": {
    "allowed": true,
    "reason": "",
    "stage": "input_guardrail"
  },
  "created_at": "2026-06-05T12:00:00Z"
}
```

Event types include `guardrail.input`, `guardrail.output`, `agent.chat.completed`, and `agent.chat.failed`.

---

## Project Structure

```
agent_builder/
├── agents/
│   ├── agent_gen.py          # Factory for creating different agent types
│   ├── loader.py             # Load agent config and create agent
│   ├── multi_agent.py        # Multi-agent graph builder + handoff tools
│   ├── reflection.py         # Reflection agent implementation
│   ├── plan_execute_replan.py
│   └── long_term_memory.py
├── core/
│   ├── interfaces.py         # Abstract adapter interfaces
│   └── types.py              # IdentityContext, AccessPolicy, etc.
├── embeddings/
│   ├── loader.py             # Load embedding models from config
│   └── adapters.py           # Concrete embedding model adapters
├── llms/
│   ├── loader.py             # Load LLMs from config
│   └── adapters.py           # Anthropic, Bedrock, Fireworks, etc.
├── tools/
│   ├── loader.py             # Load tools from config
│   ├── adapters.py           # Tool adapters (vector search, MCP, etc.)
│   ├── mongodb.py            # MongoDB vector and full-text search
│   └── mcp.py                # Model Context Protocol integration
├── memory/
│   ├── adapters.py           # MemoryAdapterFactory
│   └── mongodb_memory.py     # MongoDB episodic/observational memory
├── state/
│   └── mongodb_state.py      # Session state persistence
├── security/
│   ├── guardrails.py         # Input/output/tool guardrails
│   └── policies.py           # Policy providers (static, MongoDB)
├── audit/
│   └── mongodb_audit.py      # Audit event logging
├── app.py                    # Flask app, /chat /health /reset /threads routes
├── cli.py                    # CLI entrypoint (agent-builder serve)
├── wsgi.py                   # Gunicorn WSGI entrypoint
├── yaml_loader.py            # YAML config parser with env var resolution
└── utils/
    ├── checkpointer.py       # MongoDB LangGraph checkpointer
    └── logging_config.py     # Structured logging with connection string sanitization
```

---

## Examples

The `examples/` directory contains ready-to-run configs:

| File | Description |
|------|-------------|
| `react_rag_mongodb.yaml` | Single ReAct agent with vector search |
| `tool_call_mcp_agent.yaml` | Tool-calling agent backed by an MCP server |
| `reflection_quality_reviewer.yaml` | Reflection agent that reviews its own answers |
| `plan_execute_replan_research.yaml` | Planning agent that researches a topic |
| `long_term_memory_assistant.yaml` | Agent with episodic memory that recalls prior conversations |
| `governed_enterprise_support.yaml` | Agent with governance, policies, and audit enabled |
| `multi_agent_customer_support.yaml` | 3-agent triage system with bidirectional handoffs |

Run any example:

```bash
# Copy the example to config/
cp examples/multi_agent_customer_support.yaml config/agents.yaml

# Edit to point to your LLM and MongoDB
# Then run:
agent-builder serve --config config/agents.yaml
# or:
make run
```

---

## Testing

```bash
# Install dev dependencies (if not already)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_multi_agent.py -v

# Run with coverage
pytest tests/ --cov=agent_builder --cov-report=html

# Via Makefile
make test
```

---

## Development

```bash
# Format code
make format
# or: black agent_builder tests && isort agent_builder tests

# Lint
make lint
# or: flake8 agent_builder && ruff check agent_builder

# Type check
mypy agent_builder --ignore-missing-imports

# Security scanning
bandit -r agent_builder/
```

---

## License

Licensed under the Apache License 2.0. See the LICENSE file for details.

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted (`make test && make lint`)
5. Submit a pull request with a clear description

For questions or issues, open a GitHub issue or check the `examples/` directory for working configs.