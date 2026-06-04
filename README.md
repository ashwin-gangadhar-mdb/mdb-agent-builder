# MAAP Agent Builder

A flexible framework for building and deploying LLM agents with various capabilities using LangChain and LangGraph.

## Overview

MAAP Agent Builder is a modular framework that allows you to configure and deploy different types of agents with various capabilities:

- **Multiple Agent Types**: Support for React, Tool-Call, Reflection, Plan-Execute-Replan, and Long-Term Memory agents
- **Diverse LLM Providers**: Integration with Anthropic, Bedrock, Fireworks, Together AI, Cohere, Azure, Ollama, and more
- **Embedding Model Support**: Bedrock, SageMaker, VertexAI, Azure, Together, Fireworks, Cohere, VoyageAI, Ollama, and HuggingFace
- **Tool Integration**: Easy-to-configure tools for extending agent capabilities
- **Thread-Based Conversations**: Support for multiple concurrent threads with independent conversation histories
- **Stateful Sessions**: Conversation history and checkpointing mechanisms
- **Web API**: Built-in Flask server for easy deployment and interaction

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/maap-agent-builder.git
cd maap-agent-builder

# Set up environment and install
make setup-env
source .venv/bin/activate
make install-and-config

# Create an agent interactively
make create-agent

# Add a tool to your agent
make add-tool

# Run your agent
make run
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/maap-agent-builder.git
cd maap-agent-builder
pip install -e .
```

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
```

Alternatively, use the provided Makefile:

```bash
make install      # For regular installation
make dev          # For development installation
make verify       # Verify your installation is working correctly
```

### Project Structure

The project uses a modern Python packaging structure with `pyproject.toml`:

- **Core Dependencies**: All main dependencies are defined in `pyproject.toml`
- **Development Dependencies**: Available as optional extras via `[dev]`
- **Configuration**: Tool configurations for black, isort, mypy, pytest, and ruff are included

### Development Setup

For a complete development environment setup:

```bash
# Create and activate a virtual environment
make setup-env
source .venv/bin/activate

# Install the package with development dependencies
make install

# Create default configuration directories
make create-config
```

### Agent Configuration Setup

The project includes several helpful Makefile targets to create and manage agent configurations:

```bash
# Create a new agent configuration with an interactive wizard
make create-agent

# Validate an existing configuration
make validate-config

# Add a new tool to your configuration
make add-tool
```

These commands provide interactive prompts to help you create properly structured configuration files without having to manually edit YAML.

## Configuration

MAAP Agent Builder is configured through YAML files and environment variables.

### Environment Variables

Create a `.env` file in the root directory with your API keys and configuration:

```
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
FIREWORKS_API_KEY=your_fireworks_key
TOGETHER_API_KEY=your_together_key
COHERE_API_KEY=your_cohere_key

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com

# Vector DB Configuration (if using long-term memory agents)
MONGODB_URI=your_mongodb_connection_string

# Application Configuration
LOG_LEVEL=INFO
FLASK_SECRET_KEY=your_flask_secret_key
```

### Agent Configuration (agents.yaml)

Create a YAML configuration file to define your agents, LLMs, embedding models, and tools:

```yaml
# Configure the embedding model
embeddings:
 - name: all-mpnet-v2
   provider: huggingface
   model_name: sentence-transformers/all-mpnet-base-v2
   normalize: true

# Configure the language model
llms:
  - name: fireworks_llm_maverick
    provider: fireworks
    model_name: accounts/fireworks/models/llama4-maverick-instruct-basic
    temperature: 0.7
    max_tokens: 4000
    streaming: False
    additional_kwargs:
      top_p: 0.9
      top_k: 50

# Configure agent tools
tools:
  - name: product_recommender
    tool_type: vector_search
    description: Searches for relevant documents in the vector store
    connection_str: ${MONGODB_URI:-mongodb://localhost:27017}
    namespace: amazon.products
    embedding_model: all-mpnet-v2  # Reference to the embedding model defined above
    additional_kwargs:
      index_name: default
      embedding_field: embedding
      text_field: text
      top_k: 5
      min_score: 0.7

# Configure checkpointing
checkpointer:
  connection_str: ${MONGODB_URI:-mongodb://localhost:27017}
  db_name: agent_state
  collection_name: checkpoints
  name: rag_agent_checkpointer

# Optional MongoDB-backed governance controls
governance:
  enabled: true
  connection_str: ${MONGODB_URI:-mongodb://localhost:27017}
  db_name: agent_control_plane
  default_policy:
    permissions:
      - "*"
  policy:
    provider: mongodb
    collection_name: agent_policies
  audit:
    enabled: true
    collection_name: agent_audit_events
  state:
    enabled: true
    collection_name: agent_sessions

# Configure the agent
agent:
  name: rag_react_agent
  agent_type: react
  llm: fireworks_llm_maverick  # Reference to the LLM defined above
  tools:
    - product_recommender  # Reference to the tool defined above
  system_prompt_path: ./prompts/rag_system_prompt.txt
```

You can create this configuration manually or use the provided Makefile targets:

```bash
# Create the basic configuration structure
make create-config

# Add a new agent interactively
make create-agent  # You'll be prompted for agent name, type, LLM provider, and model

# Add a new tool interactively
make add-tool  # You'll be prompted for tool name, type, and description
```

## API Usage

The MAAP Agent Builder provides a REST API for interacting with your agents. Here are the main endpoints:

### Chat Endpoint

```
POST /chat
```

Request body:

```json
{
  "message": "Your user message here",
  "config": {
    "thread_id": "optional-thread-id"  // If not provided, a new thread will be created
  }
}
```

Response:

```json
{
  "response": "Agent's response",
  "history": [
    ["user", "Your user message here"],
    ["assistant", "Agent's response"]
  ],
  "thread_id": "thread-id"  // Use this ID to continue the conversation
}
```

### Reset Conversation History

```
POST /reset
```

Request body:

```json
{
  "thread_id": "thread-id"  // Optional. If not provided, all threads will be reset
}
```

Response:

```json
{
  "status": "success",
  "message": "Chat history reset for thread thread-id"
}
```

### List Active Threads

```
GET /threads
```

Response:

```json
{
  "status": "success",
  "threads": ["thread-id-1", "thread-id-2"],
  "count": 2
}
```

### Health Check

```
GET /health
```

Response:

```json
{
  "status": "healthy",
  "agent_loaded": true
}
```

## Thread-Based Conversation Management

MAAP Agent Builder uses thread-based conversation history to support multiple concurrent conversations:

- Each conversation is assigned a unique `thread_id`
- You can specify your own `thread_id` or let the system generate one
- The conversation history is maintained separately for each thread
- You can reset a specific thread or all threads using the `/reset` endpoint
- Thread IDs can be used to implement multi-user support or to separate different conversation contexts

## Command-Line Interface

MAAP Agent Builder provides a CLI for easy server management:

```bash
# Start the server with default settings
agent-builder serve --config config/agents.yaml

# Start with custom host and port
agent-builder serve --config config/agents.yaml --host 127.0.0.1 --port 8000

# Run in debug mode with verbose logging
agent-builder serve --config config/agents.yaml --debug --log-level DEBUG

# Load environment variables from a specific file
agent-builder serve --config config/agents.yaml --env-file .env.production
```

You can also use the provided Makefile target:

```bash
make run  # Uses the default configuration at config/agents.yaml
```

## Docker Support

You can run MAAP Agent Builder in Docker for easy deployment:

```bash
# Build the Docker image
make docker-build

# Run the Docker container
make docker-run
```

## Docker Support

You can run MAAP Agent Builder in Docker for easy deployment:

```bash
# Build the Docker image
make docker-build

# Run the Docker container
make docker-run
```

The Docker container mounts your local configuration files, logs, and prompts directories, so you can modify them without rebuilding the image.

MAAP Agent Builder supports several agent types, each with different capabilities:

1. **react**: ReAct agents that think step-by-step and use tools
2. **tool_call**: Agents that use OpenAI-style tool calling
3. **reflect**: Agents that use a generate-reflect loop for improved reasoning
4. **plan_execute_replan**: Agents that plan, execute steps, and replan as needed
5. **long_term_memory**: Agents with vector store-backed long-term memory

### Agent Type-Specific Configuration

Different agent types require different configuration parameters:

#### React Agent
```yaml
agent:
  agent_type: react
  name: react_agent
  llm: gpt4
  system_prompt: "You are a helpful assistant..."
  tools:
    - search_tool
```

#### Reflection Agent
```yaml
agent:
  agent_type: reflect
  name: reflection_agent
  llm: claude
  system_prompt: "You are a helpful assistant..."
  reflection_prompt: "Review your previous response and improve it..."
  tools:
    - calculator
```

#### Plan-Execute-Replan Agent
```yaml
agent:
  agent_type: plan_execute_replan
  name: planner_agent
  llm: gpt4
  system_prompt: "You are a helpful assistant..."
  tools:
    - search_tool
    - calculator
```

#### Long-Term Memory Agent
```yaml
agent:
  agent_type: long_term_memory
  name: memory_agent
  llm: claude
  connection_str: ${MONGODB_URI}
  namespace: agent_db.memories
  tools:
    - search_tool
```

## Running Locally

There are several ways to run the MAAP Agent Builder locally:

### 1. Using the CLI

```bash
# Set the configuration path
export AGENT_CONFIG_PATH=/path/to/your/agents.yaml

# Run the server
python -m agent_builder.cli serve --config /path/to/your/agents.yaml --port 5000
```

### 2. Using WSGI

```bash
# Set the configuration path
export AGENT_CONFIG_PATH=/path/to/your/agents.yaml

# Run with Gunicorn (recommended for production)
gunicorn -b 0.0.0.0:5000 agent_builder.wsgi:application
```

### 3. Using Python Directly

```python
from agent_builder.app import AgentApp

# Create the agent app with your configuration
agent_app = AgentApp('/path/to/your/agents.yaml')

# Run the app
agent_app.run(host='0.0.0.0', port=5000, debug=True)
```

### 4. Using Docker

The project includes Docker support for easy deployment:

```bash
# Build the Docker image
make docker-build

# Run the Docker container
make docker-run

# For debugging with an interactive shell
make docker-debug
```

When running with Docker, environment variables from your `.env` file are automatically passed to the container. Additional environment variables can be passed at runtime:

```bash
# Pass specific environment variables to the Docker container
LOG_LEVEL=DEBUG make docker-run
```

## API Endpoints

Once the server is running, you can interact with your agent through the following endpoints:

- **GET /health**: Health check endpoint
- **POST /chat**: Send a message to the agent
  ```json
  {
    "message": "What is the capital of France?"
  }
  ```
- **POST /reset**: Reset the conversation history

## Example: Curl Commands

```bash
# Health check
curl http://localhost:5000/health

# Send a message to the agent
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'

# Reset conversation
curl -X POST http://localhost:5000/reset
```

## Advanced Configuration

### Loading Prompts from Files

Instead of including prompts directly in the YAML, you can load them from files:

```yaml
agent:
  agent_type: react
  name: my_agent
  llm: gpt4
  system_prompt_path: /path/to/system_prompt.txt
  tools:
    - search_tool
```

### Environment Variable Substitution

The configuration supports environment variable substitution with default values:

```yaml
llms:
  - name: openai_llm
    provider: openai
    model_name: ${OPENAI_MODEL_NAME:-gpt-4-turbo}
    temperature: ${TEMPERATURE:-0.7}
```

### MongoDB Checkpointing

For persistent conversations across restarts, configure a MongoDB checkpointer:

```yaml
checkpointer:
  connection_str: ${MONGODB_CONNECTION_STRING}
  db_name: langgraph
  collection_name: checkpoints
```

## Project Structure

The MAAP Agent Builder is organized into several modules:

```
agent_builder/
├── agents/            # Agent implementations (React, ReflexionAgent, etc.)
├── config/            # Configuration loading and processing
├── embeddings/        # Embedding model implementations
├── llms/              # LLM provider integrations
├── tools/             # Tool implementations
└── utils/             # Utility functions and helpers
```

### Key Files

- `pyproject.toml`: Defines project metadata, dependencies, and tool configurations
- `agent_builder/app.py`: The main Flask application
- `agent_builder/cli.py`: Command-line interface
- `agent_builder/yaml_loader.py`: YAML configuration processor
- `agent_builder/agents.yaml`: Default agent configuration

### Development Workflow

1. Install the package with development dependencies
2. Make changes to the codebase
3. Run linting and tests to verify your changes
4. Build and test with Docker if needed

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set in your environment variables
2. **Configuration Loading Error**: Check your YAML syntax for errors
   - Use `make validate-config` to verify your agents.yaml file
3. **LLM Provider Not Found**: Verify that the LLM provider is supported and correctly configured
4. **Tool Execution Failed**: Check that tools have all required parameters
5. **Installation Issues**: If you encounter installation problems:
   - Ensure you have the latest pip version: `pip install --upgrade pip`
   - Try installing with verbose output: `pip install -e ".[dev]" -v`
   - Check for conflicting dependencies in your environment

### Logging

Adjust the logging level to get more detailed information:

```bash
export LOG_LEVEL=DEBUG
```

### Package Development

When developing the package:

```bash
# Run linting checks
make lint

# Format code automatically
make format

# Run tests
make test

# Clean build artifacts
make clean

# Build distribution packages
make build-package
```

### Agent Development

When developing agents and tools:

```bash
# Create a new agent configuration
make create-agent

# Add a new tool to your configuration
make add-tool

# Validate your configuration
make validate-config

# Run your agent locally
make run

# Run in Docker container
make docker-run
```

## Contributing

Contributions to MAAP Agent Builder are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

1. **Fork the repository** and clone it locally
2. **Create a new branch** for your feature or bugfix
3. **Make your changes** and ensure tests pass
4. **Run linting** to ensure code quality: `make lint`
5. **Add tests** for new functionality
6. **Submit a pull request** with a clear description of your changes

### Development Guidelines

- Follow PEP 8 style guidelines
- Write docstrings for functions and classes
- Add type hints to new code
- Ensure test coverage for new features

### Testing

Run the test suite with:

```bash
make test
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
