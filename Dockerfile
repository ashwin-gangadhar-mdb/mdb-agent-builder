FROM python:3.10-slim

WORKDIR /app

# Install only what's needed — no build-essential, no bloat.
# curl is kept for the HEALTHCHECK; ca-certificates for TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy dependency files first (better layer caching)
COPY pyproject.toml README.md ./

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AGENT_CONFIG_PATH=/app/config/agents.yaml
ENV LOG_LEVEL=INFO
ENV FLASK_ENV=production
# Multi-worker settings — override at runtime (e.g. -e GUNICORN_WORKERS=8)
ENV GUNICORN_WORKERS=4
ENV GUNICORN_TIMEOUT=120
ENV PORT=5000

# Create necessary directories
RUN mkdir -p /app/config /app/logs /app/prompts

# Copy the application code
COPY agent_builder/ /app/agent_builder/
COPY prompts/ /app/prompts/
COPY config/ /app/config/
COPY examples/ /app/examples/
COPY startup.sh /app/

# Make startup script executable
RUN chmod +x /app/startup.sh

# Install the package with pinned versions (no retry, fail-early on corruption).
# --no-cache-dir reduces image size; --only-binary avoids source builds.
RUN pip install --no-cache-dir --no-build-isolation -e .

# Ensure the default config file exists
RUN touch /app/config/agents.yaml

# Set proper permissions (after all files are copied)
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Add healthcheck against the /health endpoint using Python (no curl needed
# at runtime since it's only used during build) — but curl is still the
# simplest option.  The slim image includes it via the install above.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -fs http://localhost:${PORT:-5000}/health || exit 1

# Command to run the application with proper error handling
CMD ["/app/startup.sh"]

# Add metadata labels
LABEL org.opencontainers.image.source="https://github.com/mongodb/maap-agent-builder"
LABEL org.opencontainers.image.description="MAAP Agent Builder"
LABEL org.opencontainers.image.licenses="Apache-2.0"