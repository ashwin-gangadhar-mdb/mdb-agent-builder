#!/bin/bash
# startup.sh - Handles application startup with proper initialization and error handling

# Ensure configuration directory exists
mkdir -p /app/config /app/logs

# Check if agents.yaml exists, create a basic one if not
if [ ! -s /app/config/agents.yaml ]; then
    echo "Warning: No agents.yaml found or file is empty. Creating minimal configuration."
    cat > /app/config/agents.yaml << EOF
# Default minimal configuration
agent:
  type: react
  llm:
    type: openai
    model: gpt-4-turbo
EOF
fi

# Number of Gunicorn workers.
# For multi-worker state sharing, configure a state: section in agents.yaml
# (or enable governance) so workers read/write history from MongoDB.
WORKERS=${GUNICORN_WORKERS:-4}
TIMEOUT=${GUNICORN_TIMEOUT:-120}
PORT=${PORT:-5000}

echo "Starting MAAP Agent Builder with ${WORKERS} worker(s) on port ${PORT}..."
exec gunicorn \
    --workers "${WORKERS}" \
    --bind "0.0.0.0:${PORT}" \
    --timeout "${TIMEOUT}" \
    --access-logfile - \
    --error-logfile - \
    agent_builder.wsgi:application
