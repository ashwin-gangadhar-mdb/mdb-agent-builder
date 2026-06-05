"""
Flask application for MAAP Agent Builder.

This module provides the main web application for the MAAP Agent Builder,
handling agent initialization, request routing, and chat history management.
"""

import os
import secrets
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is an optional convenience dependency
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from flask import Flask, jsonify, request, g

from agent_builder.audit.mongodb_audit import MongoDBAuditProvider
from agent_builder.core.interfaces import BaseAuditAdapter, BasePolicyAdapter, BaseStateAdapter
from agent_builder.core.types import AuditEvent, IdentityContext
from agent_builder.security.guardrails import GuardrailEngine
from agent_builder.security.policies import MongoDBPolicyProvider, StaticPolicyProvider
from agent_builder.state.mongodb_state import MongoDBStateProvider
from agent_builder.utils.logging_config import get_logger
from agent_builder.yaml_loader import load_application

# Load environment variables from .env file if present
load_dotenv()

# Initialize logger
logger = get_logger(__name__)


class AgentApp:
    """
    Flask application for serving agents loaded from YAML configurations.
    """

    def __init__(
        self,
        config_path: str,
        session_ttl: int = 3600,
        max_history_messages: int = 100,
        max_threads: int = 1000,
    ):
        """
        Initialize the agent application with the specified YAML configuration.

        Args:
            config_path: Path to the YAML configuration file
            session_ttl: Time-to-live for session data in seconds (default: 1 hour)
            max_history_messages: Cap on retained messages per thread in the
                process-local history (prevents unbounded growth).
            max_threads: Cap on the number of threads held in process-local
                history before the oldest is evicted.

        Note:
            ``chat_histories`` is **process-local** and is not shared across
            multiple gunicorn workers.  Configure a ``state`` provider or a
            LangGraph checkpointer for durable, cross-worker conversation state.
        """
        self.app = Flask(__name__)
        secret_key = os.environ.get("FLASK_SECRET_KEY")
        if not secret_key:
            if os.environ.get("FLASK_ENV") == "production":
                raise ValueError("FLASK_SECRET_KEY must be set in production")
            secret_key = secrets.token_hex(32)
            logger.warning(
                "Auto-generated FLASK_SECRET_KEY; sessions will be "
                "invalidated on restart. Set FLASK_SECRET_KEY explicitly."
            )
        self.app.secret_key = secret_key
        self.app.config["PERMANENT_SESSION_LIFETIME"] = session_ttl
        self.app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request limit
        self.config_path = config_path
        self.components = None
        self.agent = None
        self.chat_histories = {}  # Process-local chat histories keyed by thread_id
        self.max_history_messages = max_history_messages
        self.max_threads = max_threads
        # Set once the agent is loaded; controls whether the app forwards its
        # own history or lets the agent's checkpointer manage conversation state.
        self._agent_has_checkpointer = False
        self.governance_enabled = False
        self.guardrails = GuardrailEngine()
        # Typed against the adapter interfaces — concrete backends are injected
        # in configure_governance() so callers remain backend-agnostic.
        self.policy_provider: BasePolicyAdapter = StaticPolicyProvider()
        self.audit_provider: Optional[BaseAuditAdapter] = None
        self.state_provider: Optional[BaseStateAdapter] = None

        # Register routes
        self.register_routes()

        # Register security middleware (after-request headers)
        self._register_security_middleware()

        # Load agent components
        self.load_components()

    def load_components(self):
        """Load agent and related components from the YAML configuration."""
        try:
            logger.info("Loading application components from %s", self.config_path)
            self.components = load_application(self.config_path)

            if "agent" not in self.components:
                logger.error("No agent configured in the YAML file")
                raise ValueError("No agent configured in the YAML file")

            self.agent = self.components.get("agent")
            if not self.agent:
                logger.error("Agent object not properly initialized")
                raise ValueError("Failed to initialize agent")

            # If the agent persists its own state via a checkpointer, the app
            # should not also forward its history (which would duplicate turns
            # in the checkpointed state).
            self._agent_has_checkpointer = bool(
                getattr(self.agent, "checkpointer", None)
            )

            self.configure_governance()

            logger.info("Agent successfully loaded from %s", self.config_path)
        except Exception as e:
            logger.error("Failed to load application components: %s", str(e))
            raise

    def configure_governance(self):
        """Configure optional MongoDB-backed policy, audit, and state providers."""
        governance_config = (self.components or {}).get("governance_config", {}) or {}
        self.governance_enabled = bool(governance_config.get("enabled", False))

        if self.governance_enabled:
            connection_str = governance_config.get("connection_str")
            db_name = governance_config.get("db_name", "agent_control_plane")
            default_policy = governance_config.get("default_policy", {"permissions": ["*"]})

            policy_config = governance_config.get("policy", {}) or {}
            if connection_str and policy_config.get("provider", "mongodb") == "mongodb":
                self.policy_provider = MongoDBPolicyProvider(
                    connection_str=connection_str,
                    db_name=policy_config.get("db_name", db_name),
                    collection_name=policy_config.get("collection_name", "agent_policies"),
                    default_policy=default_policy,
                )
            else:
                self.policy_provider = StaticPolicyProvider(default_policy)

            audit_config = governance_config.get("audit", {}) or {}
            if connection_str and audit_config.get("enabled", True):
                self.audit_provider = MongoDBAuditProvider(
                    connection_str=connection_str,
                    db_name=audit_config.get("db_name", db_name),
                    collection_name=audit_config.get("collection_name", "agent_audit_events"),
                )

            state_config = governance_config.get("state", {}) or {}
            if connection_str and state_config.get("enabled", True):
                self.state_provider = MongoDBStateProvider(
                    connection_str=connection_str,
                    db_name=state_config.get("db_name", db_name),
                    collection_name=state_config.get("collection_name", "agent_sessions"),
                )

            logger.info("Governance controls enabled")
        else:
            logger.info("Governance controls disabled")

        # Standalone state config — activated independently of governance, enabling
        # cross-worker session state without requiring the full governance stack.
        if self.state_provider is None:
            state_standalone = (self.components or {}).get("state_config", {}) or {}
            standalone_conn = state_standalone.get("connection_str")
            if standalone_conn and state_standalone.get("enabled", True):
                self.state_provider = MongoDBStateProvider(
                    connection_str=standalone_conn,
                    db_name=state_standalone.get("db_name", "agent_state"),
                    collection_name=state_standalone.get(
                        "collection_name", "agent_sessions"
                    ),
                )
                logger.info("Standalone state provider configured for multi-worker deployments")

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_response(message: str, status: int, **extra) -> tuple:
        """Return a JSON error response without exposing internals."""
        return jsonify({"error": message, **extra}), status

    def _check_thread_ownership(self, thread_id: str, identity: IdentityContext):
        """Return True if *identity* owns *thread_id* or the thread is unknown.

        An unknown thread is considered safe — it does not exist yet.
        When a state provider is configured with tenant isolation, a
        thread that belongs to a *different* tenant is rejected.
        """
        if not self.state_provider:
            return True
        try:
            doc = self.state_provider.load_thread(thread_id)
        except Exception:
            logger.warning("Failed to verify thread ownership for %s", thread_id)
            return True
        if doc is None:
            return True
        return doc.get("tenant_id", "default") == identity.tenant_id

    def _require_csrf_header(self):
        """Decorator that requires ``X-Requested-With: XMLHttpRequest``.

        Browsers cannot set this header on cross-origin requests without
        a preflight CORS approval, providing a basic CSRF defence when the
        API is consumed from a browser-frontend.
        """

        def decorator(fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if request.headers.get("X-Requested-With") != "XMLHttpRequest":
                    return self._error_response(
                        "Missing required X-Requested-With header", 403
                    )
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Simple in-process rate limiter
    # ------------------------------------------------------------------

    _rate_limit_buckets: Dict[str, tuple] = {}

    @classmethod
    def _check_rate_limit(cls, key: str, max_requests: int, window_secs: int) -> bool:
        """Return True if the caller is within the rate limit for *key*."""
        now = time.time()
        count, reset = cls._rate_limit_buckets.get(key, (0, now + window_secs))
        if now > reset:
            count = 0
            reset = now + window_secs
        count += 1
        cls._rate_limit_buckets[key] = (count, reset)
        # Prune stale buckets periodically (every ~1000 requests)
        if len(cls._rate_limit_buckets) > 1000:
            stale = [k for k, (_, r) in cls._rate_limit_buckets.items() if now > r]
            for k in stale:
                cls._rate_limit_buckets.pop(k, None)
        return count <= max_requests

    def _register_security_middleware(self):
        """Apply after-request security headers and log masking."""

        @self.app.after_request
        def _add_security_headers(response):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "0"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
            return response

    def record_audit(
        self,
        event_type: str,
        identity: IdentityContext,
        thread_id: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ):
        """Record an audit event without failing the user request on audit errors."""
        if not self.audit_provider:
            return
        try:
            agent_name = None
            if self.components and "agent" in self.components:
                agent_name = getattr(self.components["agent"], "name", None)
            self.audit_provider.record(
                AuditEvent(
                    event_type=event_type,
                    tenant_id=identity.tenant_id,
                    user_id=identity.user_id,
                    agent_id=agent_name,
                    thread_id=thread_id,
                    payload=payload or {},
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to record audit event %s: %s", event_type, str(e))

    def _build_run_config(
        self,
        request_config: Dict[str, Any],
        thread_id: str,
        identity: IdentityContext,
    ) -> Dict[str, Any]:
        """Build a LangGraph run config.

        LangGraph checkpointers and memory-aware agents read ``thread_id`` and
        identity fields from ``config['configurable']`` — not from the top
        level — so we place them there explicitly while preserving any other
        run options (e.g. ``recursion_limit``) the caller supplied.
        """
        run_config = dict(request_config or {})
        run_config.pop("identity", None)
        configurable = dict(run_config.get("configurable") or {})
        configurable.setdefault("thread_id", thread_id)
        configurable.setdefault("user_id", identity.user_id)
        configurable.setdefault("tenant_id", identity.tenant_id)
        configurable.setdefault("roles", list(identity.roles))
        run_config["configurable"] = configurable
        return run_config

    def _invoke_agent(
        self,
        user_message: str,
        chat_history: list,
        extra_inputs: Dict[str, Any],
        run_config: Dict[str, Any],
    ) -> str:
        """Invoke the loaded agent and return its textual response.

        Raises whatever the agent raises; callers translate that into an HTTP
        error response rather than persisting an error string as a reply.
        """
        if hasattr(self.agent, "invoke"):
            # With a checkpointer the agent accumulates history per thread, so
            # only the new turn is sent; otherwise the app-managed history
            # supplies the conversational context.
            if self._agent_has_checkpointer:
                messages = [("user", user_message)]
            else:
                messages = chat_history + [("user", user_message)]
            input_data: Dict[str, Any] = {"messages": messages}
            for key, value in extra_inputs.items():
                if key not in ("message", "history"):
                    input_data[key] = value
            response = self.agent.invoke(input_data, config=run_config)
            return self._extract_response(response)

        # Legacy callable-agent fallback.
        logger.warning("Using legacy agent format")
        return str(self.agent(user_message))

    @staticmethod
    def _extract_response(response: Any) -> str:
        """Extract the assistant's text from a LangGraph/legacy agent response."""
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            if not messages:
                return "No response from agent"
            last_message = messages[-1]
            if isinstance(last_message, tuple) and len(last_message) >= 2:
                return last_message[1]
            if hasattr(last_message, "content"):
                return last_message.content
            return str(last_message)
        return str(response)

    def _load_history(self, thread_id: str) -> list:
        """Return chat history for *thread_id*.

        When a state provider is configured, the authoritative copy lives in
        MongoDB so every Gunicorn worker sees the same state.  The process-local
        dict is used as a fallback (e.g. when the agent has its own checkpointer
        and no state provider is configured, or when the DB is temporarily
        unavailable).
        """
        if self.state_provider:
            try:
                doc = self.state_provider.load_thread(thread_id)
                if doc:
                    raw = doc.get("state", {}).get("history", [])
                    # MongoDB round-trips tuples as lists; restore tuple form so
                    # the rest of the code can rely on a consistent type.
                    return [tuple(m) if isinstance(m, list) else m for m in raw]
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Failed to load thread state from provider: %s", e)
        return list(self.chat_histories.get(thread_id, []))

    def _store_history(self, thread_id: str, history: list) -> list:
        """Persist process-local history for *thread_id* with bounded growth."""
        if self.max_history_messages and len(history) > self.max_history_messages:
            history = history[-self.max_history_messages:]
        # Evict the oldest thread when the cap is exceeded (insertion-order FIFO).
        if (
            thread_id not in self.chat_histories
            and self.max_threads
            and len(self.chat_histories) >= self.max_threads
        ):
            oldest = next(iter(self.chat_histories))
            self.chat_histories.pop(oldest, None)
            logger.info(
                "Evicted oldest in-memory thread %s (cap=%d)",
                oldest,
                self.max_threads,
            )
        self.chat_histories[thread_id] = history
        return history

    def register_routes(self):
        """Register API routes for the Flask application."""

        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            if self.agent:
                return jsonify({"status": "healthy", "agent_loaded": True})
            return jsonify({"status": "unhealthy", "agent_loaded": False}), 503

        @self.app.route("/chat", methods=["POST"])
        def chat():
            """Chat endpoint to interact with the agent."""
            if not self.agent:
                return self._error_response("Agent not loaded", 503)

            if not request.is_json:
                return self._error_response("Content-Type must be application/json", 415)

            data = request.get_json(silent=True)
            if not data or "message" not in data:
                return self._error_response("Missing required field: message", 400)

            # Work on a shallow copy so the parsed request body is never mutated.
            data = dict(data)
            request_config = data.pop("config", None) or {}
            identity = IdentityContext.from_config(request_config)
            thread_id = request_config.get("thread_id") or str(uuid.uuid4())

            # ── rate limit ────────────────────────────────────────────────────
            rate_key = f"chat:{identity.tenant_id}:{identity.user_id}"
            if not self._check_rate_limit(rate_key, max_requests=60, window_secs=60):
                return self._error_response("Too many requests — rate limit exceeded", 429)

            # ── thread ownership check ────────────────────────────────────────
            if not self._check_thread_ownership(thread_id, identity):
                return self._error_response("Thread not found", 404, thread_id=thread_id)

            user_message = data.pop("message")
            policy = self.policy_provider.get_policy(identity)

            # ── input guardrail ──────────────────────────────────────────────
            if self.governance_enabled:
                decision = self.guardrails.check_input(user_message, policy)
                self.record_audit(
                    "guardrail.input",
                    identity,
                    thread_id,
                    {
                        "allowed": decision.allowed,
                        "reason": decision.reason,
                        "stage": decision.stage,
                    },
                )
                if not decision.allowed:
                    return self._error_response(
                        "Request blocked by input guardrail",
                        403,
                        reason=decision.reason,
                        thread_id=thread_id,
                    )
                user_message = decision.transformed_text or user_message

            logger.info(
                "Received chat request: %s... for thread %s",
                user_message[:50],
                thread_id,
            )

            chat_history = self._load_history(thread_id)
            run_config = self._build_run_config(request_config, thread_id, identity)

            # ── invoke the agent ─────────────────────────────────────────────
            try:
                agent_response = self._invoke_agent(
                    user_message, chat_history, data, run_config
                )
            except Exception as agent_error:  # pylint: disable=broad-except
                logger.exception("Error in agent invocation: %s", agent_error)
                self.record_audit(
                    "agent.chat.failed", identity, thread_id, {"error": str(agent_error)}
                )
                return self._error_response(
                    "Agent invocation failed", 500, thread_id=thread_id
                )

            # ── output guardrail ─────────────────────────────────────────────
            if self.governance_enabled:
                decision = self.guardrails.check_output(agent_response, policy)
                self.record_audit(
                    "guardrail.output",
                    identity,
                    thread_id,
                    {
                        "allowed": decision.allowed,
                        "reason": decision.reason,
                        "stage": decision.stage,
                    },
                )
                if not decision.allowed:
                    return self._error_response(
                        "Response blocked by output guardrail",
                        403,
                        reason=decision.reason,
                        thread_id=thread_id,
                    )
                agent_response = decision.transformed_text or agent_response

            # ── persist history + respond ────────────────────────────────────
            updated_history = chat_history + [
                ("user", user_message),
                ("assistant", agent_response),
            ]
            updated_history = self._store_history(thread_id, updated_history)
            if self.state_provider:
                try:
                    self.state_provider.save_thread(
                        thread_id,
                        {"history": updated_history},
                        tenant_id=identity.tenant_id,
                        user_id=identity.user_id,
                    )
                except Exception as state_error:  # pylint: disable=broad-except
                    logger.warning("Failed to persist thread state: %s", state_error)
            self.record_audit(
                "agent.chat.completed",
                identity,
                thread_id,
                {"history_length": len(updated_history)},
            )

            return jsonify(
                {
                    "response": agent_response,
                    "history": updated_history,
                    "thread_id": thread_id,
                }
            )

        @self.app.route("/reset", methods=["POST"])
        @self._require_csrf_header()
        def reset():
            """Reset the chat history for a specific thread or all threads."""
            try:
                data = request.json or {}
                thread_id = data.get("thread_id")

                if thread_id:
                    identity_raw = data.get("identity", {})
                    tenant_id = identity_raw.get("tenant_id", "default")
                    if not self._check_thread_ownership(
                        thread_id,
                        IdentityContext(tenant_id=tenant_id, user_id="admin"),
                    ):
                        return self._error_response(
                            "Thread not found", 404, thread_id=thread_id
                        )
                    if thread_id in self.chat_histories:
                        self.chat_histories[thread_id] = []
                        logger.info("Reset chat history for thread %s", thread_id)
                        return jsonify(
                            {
                                "status": "success",
                                "message": f"Chat history reset for thread {thread_id}",
                            }
                        )
                    else:
                        logger.warning(
                            "Attempted to reset non-existent thread %s", thread_id
                        )
                        return jsonify(
                            {
                                "status": "warning",
                                "message": f"Thread {thread_id} not found",
                            }
                        )
                else:
                    self.chat_histories = {}
                    logger.info("Reset all chat histories")
                    return jsonify(
                        {"status": "success", "message": "All chat histories reset"}
                    )
            except Exception:
                logger.exception("Error resetting chat history")
                return self._error_response("Failed to reset chat history", 500)

        @self.app.route("/threads", methods=["GET"])
        def list_threads():
            """List all active thread IDs."""
            try:
                threads = list(self.chat_histories.keys())
                return jsonify(
                    {"status": "success", "threads": threads, "count": len(threads)}
                )
            except Exception:
                logger.exception("Error listing threads")
                return self._error_response("Failed to list threads", 500)

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the Flask application."""
        logger.info("Starting agent server on %s:%s", host, port)
        self.app.run(host=host, port=port, debug=debug)


def create_app(config_path: str) -> Flask:
    """
    Factory function to create and configure a Flask application.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configured Flask application
    """
    agent_app = AgentApp(config_path)
    return agent_app.app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the MAAP Agent Builder Flask application"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    agent_app = AgentApp(args.config)
    agent_app.run(host=args.host, port=args.port, debug=args.debug)
