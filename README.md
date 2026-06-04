# MAAP Agent Builder

A production-ready framework for building and deploying LLM agents with multiple agent types, multi-agent coordination, advanced memory systems, and governance controls. Built on LangChain, LangGraph, and MongoDB.

## Overview

MAAP Agent Builder enables you to:

- **Build diverse agent types** — ReAct, Tool-Call, Reflection, Plan-Execute-Replan, and Long-Term Memory agents, all via YAML
- **Coordinate multiple agents** — configure handoffs between agents so they can route to each other based on conversation context
- **Scale across workers** — Gunicorn multi-worker support with MongoDB-backed conversation state and checkpointing
- **Add long-term memory** — episodic (verbatim) and observational (distilled) memory with MongoDB Atlas Vector Search
- **Enforce governance** — access policies, prompt injection detection, PII redaction, and audit logging
- **Integrate any LLM or tool** — pluggable adapters for Anthropic, Bedrock, Fireworks, Cohere, and 10+ other providers
- **Persist across restarts** — MongoDB checkpointing for durable graph and conversation state

## Quick Start

```bash
git clone https://github.com/mongodb/maap-agent-builder.git
cd maap-agent-builder

pip install -e .

# Create a basic agent config (or use examples/)
cp config/agents.yaml my_config.yaml
export MONGODB_URI=mongodb://localhost:27017
export ANTHROPIC_API_KEY=your_key

# Start the server with Gunicorn (4 workers by default)
agent-builder serve --config my_config.yaml

# Or use Flask dev server for development
GUNICORN_WORKERS=1 agent-builder serve --config my_config.yaml
```

Visit `http://localhost:5000/health` and start chatting at `POST /chat`.

## Installation

### From source

```bash
git clone https://github.com/mongodb/maap-agent-builder.git
cd maap-agent-builder
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t maap-agent-builder .
docker run -e MONGODB_URI=... -e ANTHROPIC_API_KEY=... \
  -e GUNICORN_WORKERS=8 \
  -p 5000:5000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/prompts:/app/prompts \
  maap-agent-builder
```

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

tools:           # Tools available to agents
  - name: search
    tool_type: vector_search
    namespace: my_db.products

memory:          # Optional: long-term memory adapters
  - name: recall
    memory_type: episodic
    embedding_model: voyage

agent:           # Single-agent mode (existing configs)
  agent_type: react
  llm: claude-3.5
  tools: [search]

agents:          # Multi-agent mode (new)
  - name: triage
    agent_type: react
    llm: claude-3.5
    handoffs:
      - name: billing
        description: "Route billing questions here"

checkpointer:    # MongoDB checkpointing (cross-worker state)
  connection_str: ${MONGODB_URI}

state:           # Standalone session state (optional)
  connection_str: ${MONGODB_URI}

governance:      # Optional: policies, audit, PII redaction
  enabled: true
  connection_str: ${MONGODB_URI}
```

### Environment Variables

```bash
# Required
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true
ANTHROPIC_API_KEY=sk-ant-...

# Optional LLM providers
OPENAI_API_KEY=sk-...
BEDROCK_REGION=us-west-2
FIREWORKS_API_KEY=...

# Grove API gateway (OpenAI-compatible LLM gateway)
GROVE_API_BASE=https://grove.example.com/v1
GROVE_API_KEY=...

# Server settings (for multi-worker deployments)
GUNICORN_WORKERS=4
GUNICORN_TIMEOUT=120
PORT=5000
LOG_LEVEL=INFO
```

### LLM Providers

Models are declared under `llms:` and referenced by name from agents. Each
provider has its own adapter; the supported `provider:` values are:

| Provider | Notes |
|----------|-------|
| `bedrock` | Amazon Bedrock chat models |
| `anthropic` | Anthropic Claude (`ANTHROPIC_API_KEY`) |
| `fireworks` | Fireworks AI (`FIREWORKS_API_KEY`) |
| `together` | Together AI (`TOGETHER_API_KEY`) |
| `cohere` | Cohere (`COHERE_API_KEY`) |
| `azure` | Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`) |
| `ollama` | Local Ollama models |
| `sagemaker` | AWS SageMaker endpoints |
| `grove` | **Grove API gateway** (OpenAI-compatible) — see below |

#### Grove API Gateway

Grove is an OpenAI-compatible LLM gateway: a single endpoint that fronts one
or more upstream model providers. Point the `grove` provider at the gateway's
base URL and the framework can use any model Grove exposes:

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

agent:
  agent_type: react
  llm: grove-claude
  tools: [search]
```

Resolution order:

- **base_url** — `additional_kwargs.base_url` → `GROVE_API_BASE` env →
  `GROVE_API_GATEWAY_URL` env. Required.
- **api_key** — `additional_kwargs.api_key` → `GROVE_API_KEY` env → a
  placeholder (for gateways that don't enforce auth).

Any other keys under `additional_kwargs` (e.g. `default_headers`,
`organization`, `timeout`) are passed straight through to the underlying
OpenAI-compatible client, so gateway-specific auth headers and options are
fully supported.

## Agent Types

### Single-Agent Configuration

Use the `agent:` key for a single agent:

```yaml
agent:
  name: my_agent
  agent_type: react        # Required: react, tool_call, reflect, plan_execute_replan, long_term_memory
  llm: claude-3.5          # Reference to an llm defined above
  system_prompt: |
    You are a helpful assistant...
  tools: [search]          # Optional: list of tool names
  
  # For reflect agents
  reflection_prompt: |
    Review and improve your previous response...

  # For long-term memory agents
  episodic_memory: my_memory    # Reference to a memory adapter
  observational_memory: analysis
```

### Multi-Agent Configuration (NEW)

Use the `agents:` key (plural) to define multiple agents that can hand off to each other:

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
    system_prompt: "Handle billing questions."
    tools: [billing_search]
    handoffs:
      - name: triage_agent

  - name: technical_agent
    agent_type: react
    llm: claude-3.5
    system_prompt: "Handle technical issues."
    tools: [product_knowledge]

entry_agent: triage_agent  # Which agent receives the first user message
```

When an agent calls a handoff tool (e.g., `transfer_to_billing_agent`), execution routes to the target agent in the same conversation thread. Use `examples/multi_agent_customer_support.yaml` as a reference.

## Multi-Worker Deployments

### Process-Local History (Single Worker)

By default, conversation history is stored in-memory. This works fine for single-worker deployments but doesn't survive process restarts.

### Cross-Worker Session State

For Gunicorn multi-worker deployments, enable either a standalone `state:` config or full `governance:` config to share conversation history across workers:

```yaml
# Option 1: Standalone state (no governance)
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
```

### Starting Multiple Workers

```bash
# 8 workers, 120-second timeout
GUNICORN_WORKERS=8 GUNICORN_TIMEOUT=120 agent-builder serve --config config/agents.yaml

# Or via Docker env var
docker run -e GUNICORN_WORKERS=8 ... maap-agent-builder
```

The `/health` endpoint responds over HTTP (no cookies or session state), so load balancers can route traffic cleanly. Each worker reads and writes chat history to the same MongoDB collection, so conversation state is consistent across the fleet.

## Memory Systems

### Episodic Memory

Stores verbatim conversation snippets (what was said, when). Useful for recall:

```yaml
memory:
  - name: episode_recall
    memory_type: episodic
    connection_str: ${MONGODB_URI}
    namespace: my_db.episodes
    embedding_model: voyage
    index_name: episodic_index

agent:
  episodic_memory: episode_recall
```

### Observational Memory

Uses an LLM to distil raw conversation into structured facts (user preferences, behavioral patterns). Useful for learning:

```yaml
memory:
  - name: observations
    memory_type: observational
    connection_str: ${MONGODB_URI}
    namespace: my_db.observations
    embedding_model: voyage
    llm: claude-3.5
    extraction_prompt: |
      Extract key facts from this conversation: {text}

agent:
  observational_memory: observations
```

## API Endpoints

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

Response:

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

### Health Check

```bash
curl http://localhost:5000/health
```

### Reset Conversation

```bash
curl -X POST http://localhost:5000/reset \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "user-123-conv-1"}'
```

### List Active Threads

```bash
curl http://localhost:5000/threads
```

## Governance & Security

### Access Policies

Define role-based or tenant-based policies in MongoDB:

```bash
db.agent_policies.insertOne({
  "tenant_id": "acme-corp",
  "role": "support",
  "permissions": ["tools.call.search", "tools.call.send_email"],
  "denied_tools": ["delete_user"],
  "blocked_topics": ["salary", "passwords"],
  "pii_redaction": true,
  "prompt_injection_detection": true
})
```

### Guardrails

Automatically applied when `governance.enabled: true`:

- **Blocked Topics** — reject requests mentioning restricted words
- **Prompt Injection Detection** — pattern-match common jailbreak attempts
- **PII Redaction** — redact emails and phone numbers from input/output
- **Tool Allowlisting** — only call tools the user's policy permits

### Audit Logging

All events are logged to MongoDB (`agent_audit_events` by default):

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
  }
}
```

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
├── app.py                    # Flask app, /chat /health /reset routes
├── yaml_loader.py            # YAML config parser
└── utils/
    ├── checkpointer.py       # MongoDB LangGraph checkpointer
    └── logging_config.py     # Structured logging
```

## Examples

The `examples/` directory contains ready-to-run configs:

- `react_rag_mongodb.yaml` — Single ReAct agent with vector search
- `reflection_quality_reviewer.yaml` — Reflection agent that reviews its own answers
- `plan_execute_replan_research.yaml` — Planning agent that researches a topic
- `long_term_memory_assistant.yaml` — Agent with episodic memory that recalls prior conversations
- `governed_enterprise_support.yaml` — Multi-agent support with governance, policies, and audit
- `multi_agent_customer_support.yaml` — 3-agent triage system with bidirectional handoffs

Run any of them:

```bash
agent-builder serve --config examples/multi_agent_customer_support.yaml
```

## Testing

```bash
pip install -e ".[dev]"

# Unit tests (no services required)
pytest tests/ -v

# Specific test file
pytest tests/test_multi_agent.py -v

# With coverage
pytest tests/ --cov=agent_builder
```

## Development

```bash
# Format code
black agent_builder tests

# Lint
flake8 agent_builder tests
mypy agent_builder

# Type check with mypy
mypy agent_builder --ignore-missing-imports
```

## License

Licensed under the Apache License 2.0. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted
5. Submit a pull request with a clear description

For questions or issues, open a GitHub issue or check the examples directory for working configs.
