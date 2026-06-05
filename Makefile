.PHONY: clean install dev test lint format docker-build docker-run docker-debug setup-env create-config create-agent validate-config add-tool run serve serve-prod build-package help verify reinstall

# Project variables
PROJECT_NAME := maap-agent-builder
PYTHON := python3
PIP := $(PYTHON) -m pip
CONFIG_DIR := config
LOGS_DIR := logs
CONFIG_PATH := $(CONFIG_DIR)/agents.yaml

# Container runtime (override on the command line, e.g. make docker-build CONTAINER_RUNTIME=podman)
CONTAINER_RUNTIME ?= docker

# Server / multi-worker settings (override on the command line, e.g.
#   make serve-prod GUNICORN_WORKERS=8 PORT=8080
#   make docker-run GUNICORN_WORKERS=8)
PORT ?= 5000
GUNICORN_WORKERS ?= 4
GUNICORN_TIMEOUT ?= 120

help:
	@echo "MAAP Agent Builder"
	@echo ""
	@echo "Usage:"
	@echo "  make setup-env       Create virtual environment and install dependencies"
	@echo "  make install         Install the package in development mode"
	@echo "  make dev             Install dev dependencies"
	@echo "  make create-config   Create default configuration directories and files"
	@echo "  make create-agent    Create a new agent configuration template"
	@echo "  make validate-config Validate the agents.yaml configuration"
	@echo "  make add-tool        Add a new tool to the agents.yaml configuration"
	@echo "  make lint            Run linting checks"
	@echo "  make format          Format code using black and isort"
	@echo "  make test            Run tests"
	@echo "  make build-package   Build package distribution files"
	@echo "  make clean           Remove build artifacts and cache directories"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container (override GUNICORN_WORKERS, PORT)"
	@echo "  make docker-debug    Start Docker container in interactive mode for debugging"
	@echo "  make run             Run the agent server (Flask dev server, single process)"
	@echo "  make serve           Alias for 'make run'"
	@echo "  make serve-prod      Run with Gunicorn (multi-worker, production-like)"
	@echo "  make verify          Verify the installation and configuration"
	@echo "  make reinstall       Reinstall the package after name changes"
	@echo ""
	@echo "Container vars (override on the command line):"
	@echo "  CONTAINER_RUNTIME=$(CONTAINER_RUNTIME)"
	@echo "Server vars (override on the command line):"
	@echo "  PORT=$(PORT)  GUNICORN_WORKERS=$(GUNICORN_WORKERS)  GUNICORN_TIMEOUT=$(GUNICORN_TIMEOUT)"

# Setup the environment and install dependencies
setup-env:
	@echo "Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created. Activate it with: source .venv/bin/activate"
	@echo "Then run: make install"
	@if [ ! -f .env ]; then \
		echo "Creating .env.example file..."; \
		if [ -f .env.example ]; then \
			echo ".env.example already exists"; \
		else \
			cp -n .env.example .env 2>/dev/null || \
			echo "Warning: Could not create .env file from example"; \
		fi; \
		echo "Copy .env.example to .env and edit with your settings"; \
	fi

# Install the package in development mode
install:
	@echo "Installing package in development mode..."
	$(PIP) install --upgrade pip setuptools wheel build
	$(PIP) install -e ".[dev]"
	@echo "Installation complete."

# Install dev dependencies (kept for backwards compatibility)
dev:
	@echo "Installing development dependencies..."
	$(PIP) install -e ".[dev]"
	@echo "Development dependencies installed."

# Create default configuration directories and files
create-config:
	@echo "Creating configuration directories..."
	mkdir -p $(CONFIG_DIR) $(LOGS_DIR) prompts
	@if [ ! -f $(CONFIG_PATH) ]; then \
		cp -n agent_builder/agents.yaml $(CONFIG_DIR)/ 2>/dev/null || \
		cp -n agent_builder/agents.yaml $(CONFIG_DIR)/ 2>/dev/null || \
		echo "Warning: Could not find default agents.yaml template"; \
		echo "Default configuration copied to $(CONFIG_PATH)"; \
	else \
		echo "Configuration file already exists at $(CONFIG_PATH)"; \
	fi
	@echo "Creating sample system prompt..."
	@if [ ! -f prompts/rag_system_prompt.txt ]; then \
		cp -n agent_builder/prompts/rag_system_prompt.txt prompts/ 2>/dev/null || \
		cp -n agent_builder/prompts/rag_system_prompt.txt prompts/ 2>/dev/null || \
		echo "You are an advanced AI assistant with tool-calling capabilities designed to provide accurate and helpful responses. You have access to various tools that you can use when appropriate.\n\nWhen responding to queries:\n\n1. DETERMINE if you need to use tools to answer accurately.\n2. CHOOSE the appropriate tool for the task.\n3. CALL the tool with clear, specific inputs.\n4. ANALYZE the tool's response carefully.\n5. FORMULATE a comprehensive answer based on the tool's output.\n\nDo not make up information. If you don't know the answer or can't find relevant information using the tools, acknowledge that.\n\nThink step-by-step when solving complex problems." > prompts/rag_system_prompt.txt; \
		echo "Sample system prompt created at prompts/rag_system_prompt.txt"; \
	else \
		echo "System prompt already exists at prompts/rag_system_prompt.txt"; \
	fi

# Create a new agent configuration template
create-agent:
	@echo "Creating new agent configuration template..."
	@mkdir -p $(CONFIG_DIR) prompts
	@if [ ! -f prompts/rag_system_prompt.txt ]; then \
		echo "Creating default system prompt file..."; \
		cp -n agent_builder/prompts/rag_system_prompt.txt prompts/ 2>/dev/null || \
		echo "You are an advanced AI assistant with tool-calling capabilities designed to provide accurate and helpful responses. You have access to various tools that you can use when appropriate.\n\nWhen responding to queries:\n\n1. DETERMINE if you need to use tools to answer accurately.\n2. CHOOSE the appropriate tool for the task.\n3. CALL the tool with clear, specific inputs.\n4. ANALYZE the tool's response carefully.\n5. FORMULATE a comprehensive answer based on the tool's output.\n\nDo not make up information. If you don't know the answer or can't find relevant information using the tools, acknowledge that.\n\nThink step-by-step when solving complex problems." > prompts/rag_system_prompt.txt; \
	fi
	@read -p "Enter agent name: " agent_name; \
	read -p "Enter agent type (react/tool_call/reflect/plan_execute_replan/long_term_memory): " agent_type; \
	read -p "Enter LLM provider (bedrock/anthropic/fireworks/together/cohere/azure/ollama/sagemaker/grove): " llm_provider; \
	read -p "Enter model name: " model_name; \
	mkdir -p $(CONFIG_DIR); \
	if [ -f $(CONFIG_PATH) ]; then \
		echo "\n# New agent configuration for $$agent_name" >> $(CONFIG_PATH); \
		if ! grep -q "^llms:" $(CONFIG_PATH); then \
			echo "llms:" >> $(CONFIG_PATH); \
		fi; \
		echo "  - name: $${agent_name}_llm" >> $(CONFIG_PATH); \
		echo "    provider: $$llm_provider" >> $(CONFIG_PATH); \
		echo "    model_name: $$model_name" >> $(CONFIG_PATH); \
		echo "    temperature: 0.7" >> $(CONFIG_PATH); \
		echo "    streaming: True" >> $(CONFIG_PATH); \
		echo "" >> $(CONFIG_PATH); \
		echo "agent:" >> $(CONFIG_PATH); \
		echo "  name: $$agent_name" >> $(CONFIG_PATH); \
		echo "  agent_type: $$agent_type" >> $(CONFIG_PATH); \
		echo "  llm: $${agent_name}_llm" >> $(CONFIG_PATH); \
		echo "  system_prompt_path: ./prompts/rag_system_prompt.txt" >> $(CONFIG_PATH); \
		echo "  tools: []" >> $(CONFIG_PATH); \
		echo "Created new agent configuration for $$agent_name in $(CONFIG_PATH)"; \
	else \
		echo "llms:" > $(CONFIG_PATH); \
		echo "  - name: $${agent_name}_llm" >> $(CONFIG_PATH); \
		echo "    provider: $$llm_provider" >> $(CONFIG_PATH); \
		echo "    model_name: $$model_name" >> $(CONFIG_PATH); \
		echo "    temperature: 0.7" >> $(CONFIG_PATH); \
		echo "    streaming: True" >> $(CONFIG_PATH); \
		echo "" >> $(CONFIG_PATH); \
		echo "agent:" >> $(CONFIG_PATH); \
		echo "  name: $$agent_name" >> $(CONFIG_PATH); \
		echo "  agent_type: $$agent_type" >> $(CONFIG_PATH); \
		echo "  llm: $${agent_name}_llm" >> $(CONFIG_PATH); \
		echo "  system_prompt_path: ./prompts/rag_system_prompt.txt" >> $(CONFIG_PATH); \
		echo "  tools: []" >> $(CONFIG_PATH); \
		echo "Created new agent configuration at $(CONFIG_PATH)"; \
	fi

# Validate the agents.yaml configuration (supports single 'agent' and
# multi-agent 'agents' configurations).
validate-config:
	@echo "Validating agent configuration..."
	@if [ -f $(CONFIG_PATH) ]; then \
		$(PYTHON) -c "import yaml; yaml.safe_load(open('$(CONFIG_PATH)'))" && echo "Configuration is valid YAML."; \
		echo "Checking for required keys..."; \
		$(PYTHON) -c "import yaml; c=yaml.safe_load(open('$(CONFIG_PATH)')); llms={l['name'] for l in c.get('llms',[])}; ags=[c['agent']] if 'agent' in c else c.get('agents',[]); assert ags,'Missing agent/agents section'; assert all('name' in a for a in ags),'An agent is missing name'; assert all('agent_type' in a for a in ags),'An agent is missing agent_type'; assert all('llm' in a for a in ags),'An agent is missing llm reference'; bad=[a.get('name','?') for a in ags if a['llm'] not in llms]; assert not bad,f'Agents reference undefined LLMs: {bad}'" && echo "Configuration contains all required fields."; \
	else \
		echo "Configuration file not found at $(CONFIG_PATH). Use 'make create-config' to create it."; \
		exit 1; \
	fi

# Run linting
lint:
	@echo "Running linting checks..."
	flake8 agent_builder
	black --check agent_builder
	isort --check-only agent_builder
	ruff check agent_builder

# Format code
format:
	@echo "Formatting code..."
	black agent_builder
	isort agent_builder
	@echo "Code formatted."

# Run tests
test:
	@echo "Running tests..."
	pytest tests/

# Clean up build artifacts and cache directories
clean:
	@echo "Cleaning build artifacts and cache directories..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf mdb_agent_builder.egg-info
	rm -rf agent_builder.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	@echo "Cleaned."

# Build package distribution files
build-package:
	@echo "Building package distribution files..."
	$(PIP) install --upgrade build
	rm -rf dist/ build/ *.egg-info
	$(PYTHON) -m build
	@echo "Package built. Distribution files available in dist/"

# Build the Docker image
docker-build:
	@echo "Building container image with $(CONTAINER_RUNTIME)..."
	$(CONTAINER_RUNTIME) build -t $(PROJECT_NAME) -f Dockerfile .
	@echo "Container image built: $(PROJECT_NAME)"

# Run the Docker container
docker-run:
	@echo "Running Docker container..."
	@if [ ! -f .env ]; then \
		echo "Warning: .env file not found. Environment variables will not be loaded."; \
		echo "You may want to copy .env.example to .env and configure it."; \
	fi
	$(CONTAINER_RUNTIME) run -p $(PORT):$(PORT) \
		-v $(PWD)/$(CONFIG_DIR):/app/config \
		-v $(PWD)/$(LOGS_DIR):/app/logs \
		-v $(PWD)/prompts:/app/prompts \
		$(if $(wildcard .env),--env-file .env,) \
		-e PYTHONPATH=/app \
		-e AGENT_CONFIG_PATH=/app/config/agents.yaml \
		-e PORT=$(PORT) \
		-e GUNICORN_WORKERS=$(GUNICORN_WORKERS) \
		-e GUNICORN_TIMEOUT=$(GUNICORN_TIMEOUT) \
		$(PROJECT_NAME)

# Run the agent server (Flask dev server — single process, for local dev)
run:
	@echo "Running MAAP Agent Builder server (dev) on port $(PORT)..."
	@if [ ! -f $(CONFIG_PATH) ]; then \
		echo "Configuration file not found at $(CONFIG_PATH). Creating default configuration..."; \
		make create-config; \
	fi
	export AGENT_CONFIG_PATH=$(CONFIG_PATH) && \
	$(PYTHON) -m agent_builder.cli serve --config $(CONFIG_PATH) --port $(PORT)

# Alias for run
serve: run

# Run with Gunicorn — multi-worker, mirrors the production / Docker path.
# Workers share conversation state via MongoDB when a 'state:' or
# 'governance:' section is configured in the YAML.
serve-prod:
	@echo "Running MAAP Agent Builder with Gunicorn ($(GUNICORN_WORKERS) workers) on port $(PORT)..."
	@if [ ! -f $(CONFIG_PATH) ]; then \
		echo "Configuration file not found at $(CONFIG_PATH). Creating default configuration..."; \
		$(MAKE) create-config; \
	fi
	AGENT_CONFIG_PATH=$(CONFIG_PATH) \
	gunicorn \
		--workers $(GUNICORN_WORKERS) \
		--bind 0.0.0.0:$(PORT) \
		--timeout $(GUNICORN_TIMEOUT) \
		--access-logfile - \
		--error-logfile - \
		agent_builder.wsgi:application

# Verify installation and configuration
verify:
	@./verify_installation.sh

# Reinstall the package after name changes
reinstall: clean
	@echo "Reinstalling package after name changes..."
	$(PIP) install --upgrade pip setuptools wheel build
	$(PIP) install -e ".[dev]"
	@echo "Package reinstalled."

# Debug the Docker container with a shell
docker-debug:
	@echo "Starting Docker container in interactive mode..."
	@if [ ! -f .env ]; then \
		echo "Warning: .env file not found. Environment variables will not be loaded."; \
	fi
	$(CONTAINER_RUNTIME) run -it --rm \
		-p $(PORT):$(PORT) \
		-v $(PWD)/$(CONFIG_DIR):/app/config \
		-v $(PWD)/$(LOGS_DIR):/app/logs \
		-v $(PWD)/prompts:/app/prompts \
		$(if $(wildcard .env),--env-file .env,) \
		-e PYTHONPATH=/app \
		-e LOG_LEVEL=DEBUG \
		-e AGENT_CONFIG_PATH=/app/config/agents.yaml \
		-e PORT=$(PORT) \
		-e GUNICORN_WORKERS=$(GUNICORN_WORKERS) \
		-e GUNICORN_TIMEOUT=$(GUNICORN_TIMEOUT) \
		$(PROJECT_NAME) /bin/bash

# Default target
.DEFAULT_GOAL := help

# Full setup (creates venv, installs package, creates config)
setup: setup-env
	@echo "Running full setup process..."
	@echo "NOTE: Please activate the virtual environment with 'source .venv/bin/activate'"
	@echo "Then run: make install-and-config"
	@if [ ! -f .env ] && [ -f .env.example ]; then \
		echo ""; \
		echo "IMPORTANT: Copy .env.example to .env and edit with your configuration:"; \
		echo "cp .env.example .env"; \
	fi

# Install and create config in one step (requires activated venv)
install-and-config: install create-config
	@echo "Installation and configuration complete."
	@if ! command -v agent-builder >/dev/null 2>&1; then \
		echo ""; \
		echo "WARNING: agent-builder command not found in PATH."; \
		echo "Make sure your virtual environment is activated."; \
	else \
		echo "agent-builder command is available."; \
	fi

# Add a tool to an existing agent configuration
add-tool:
	@echo "Adding a new tool to the agent configuration..."
	@if [ ! -f $(CONFIG_PATH) ]; then \
		echo "Configuration file not found at $(CONFIG_PATH). Use 'make create-config' to create it."; \
		exit 1; \
	fi; \
	read -p "Enter tool name: " tool_name; \
	read -p "Enter tool type (vector_search/full_text_search/mongodb_toolkit/nl_to_mql/mcp): " tool_type; \
	read -p "Enter tool description: " tool_description; \
	if grep -q "^tools:" $(CONFIG_PATH); then \
		echo "\n# New tool configuration for $$tool_name" >> $(CONFIG_PATH); \
		sed -i.bak '/^tools:/,/^[^[:space:]]/ { /^[^[:space:]]/! { /^$$/! s/$$/ \\/ } }' $(CONFIG_PATH); \
		sed -i.bak '/^tools:/a\ \ - name: '"$$tool_name"'\n    tool_type: '"$$tool_type"'\n    description: "'"$$tool_description"'"' $(CONFIG_PATH); \
		rm -f $(CONFIG_PATH).bak; \
	else \
		echo "\n# Tool configuration" >> $(CONFIG_PATH); \
		echo "tools:" >> $(CONFIG_PATH); \
		echo "  - name: $$tool_name" >> $(CONFIG_PATH); \
		echo "    tool_type: $$tool_type" >> $(CONFIG_PATH); \
		echo "    description: \"$$tool_description\"" >> $(CONFIG_PATH); \
	fi; \
	echo "" >> $(CONFIG_PATH); \
	echo "# Don't forget to add this tool to your agent's tools list:" >> $(CONFIG_PATH); \
	echo "# Update your agent configuration with:" >> $(CONFIG_PATH); \
	echo "#   tools:" >> $(CONFIG_PATH); \
	echo "#     - $$tool_name" >> $(CONFIG_PATH); \
	echo "Added new tool $$tool_name to $(CONFIG_PATH)"; \
	echo "NOTE: You need to manually add the tool to your agent's tools list."
