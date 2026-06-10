import os
import unittest

from flask import Flask

from agent_builder.app import AgentApp
from agent_builder.core.types import IdentityContext
from agent_builder.security.guardrails import GuardrailEngine
from agent_builder.security.policies import StaticPolicyProvider


class _FakeAgent:
    """LangGraph-like agent that records the config it was invoked with."""

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer
        self.last_input = None
        self.last_config = None

    def invoke(self, input_data, config=None):
        self.last_input = input_data
        self.last_config = config
        return {"messages": [("assistant", "echo")]}


class _ExplodingAgent:
    checkpointer = None

    def invoke(self, input_data, config=None):
        raise RuntimeError("boom")


def _make_app(agent):
    app = AgentApp.__new__(AgentApp)
    app.app = Flask(__name__)
    app.app.secret_key = "test"
    app.config_path = "unused"
    app.components = {"agent": agent}
    app.agent = agent
    app.chat_histories = {}
    app.max_history_messages = 100
    app.max_threads = 1000
    app._agent_has_checkpointer = bool(getattr(agent, "checkpointer", None))
    app.governance_enabled = False
    app.guardrails = GuardrailEngine()
    app.policy_provider = StaticPolicyProvider()
    app.audit_provider = None
    app.state_provider = None
    app.register_routes()
    return app


class ChatEndpointTests(unittest.TestCase):
    def test_missing_message_returns_400(self):
        client = _make_app(_FakeAgent()).app.test_client()
        resp = client.post("/chat", json={})
        self.assertEqual(resp.status_code, 400)

    def test_thread_id_wired_under_configurable(self):
        agent = _FakeAgent()
        client = _make_app(agent).app.test_client()
        resp = client.post(
            "/chat", json={"message": "hi", "config": {"thread_id": "t-123"}}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["thread_id"], "t-123")
        # The fix: LangGraph reads thread_id from config['configurable'].
        self.assertEqual(agent.last_config["configurable"]["thread_id"], "t-123")
        self.assertEqual(agent.last_config["configurable"]["user_id"], "anonymous")
        self.assertEqual(agent.last_config["configurable"]["tenant_id"], "default")

    def test_identity_propagated_into_configurable(self):
        agent = _FakeAgent()
        client = _make_app(agent).app.test_client()
        client.post(
            "/chat",
            json={
                "message": "hi",
                "config": {"thread_id": "t", "user_id": "u9", "tenant_id": "acme"},
            },
        )
        configurable = agent.last_config["configurable"]
        self.assertEqual(configurable["user_id"], "u9")
        self.assertEqual(configurable["tenant_id"], "acme")

    def test_recursion_limit_passthrough(self):
        agent = _FakeAgent()
        client = _make_app(agent).app.test_client()
        client.post(
            "/chat",
            json={"message": "hi", "config": {"thread_id": "t", "recursion_limit": 9}},
        )
        self.assertEqual(agent.last_config["recursion_limit"], 9)

    def test_agent_error_returns_500_and_does_not_store_error_as_history(self):
        app = _make_app(_ExplodingAgent())
        client = app.app.test_client()
        resp = client.post(
            "/chat", json={"message": "hi", "config": {"thread_id": "err"}}
        )
        # Regression: failures used to be stored as the assistant reply and
        # returned with HTTP 200.
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.get_json()["error"], "Agent invocation failed")
        self.assertNotIn("err", app.chat_histories)

    def test_successful_chat_records_history(self):
        app = _make_app(_FakeAgent())
        client = app.app.test_client()
        resp = client.post(
            "/chat", json={"message": "hello", "config": {"thread_id": "t"}}
        )
        history = resp.get_json()["history"]
        self.assertEqual(history[-2], ["user", "hello"])
        self.assertEqual(history[-1], ["assistant", "echo"])
        self.assertEqual(len(app.chat_histories["t"]), 2)

    def test_checkpointer_agent_gets_only_new_turn(self):
        # When the agent persists its own state, the app forwards only the new
        # turn (the checkpointer accumulates history) to avoid duplication.
        agent = _FakeAgent(checkpointer=object())
        app = _make_app(agent)
        client = app.app.test_client()
        app.chat_histories["t"] = [("user", "old"), ("assistant", "older")]
        client.post("/chat", json={"message": "new", "config": {"thread_id": "t"}})
        self.assertEqual(agent.last_input["messages"], [("user", "new")])


class RunConfigHelperTests(unittest.TestCase):
    def setUp(self):
        self.app = AgentApp.__new__(AgentApp)

    def test_build_run_config_places_identity_in_configurable(self):
        identity = IdentityContext(tenant_id="t1", user_id="u1", roles=["admin"])
        run_config = self.app._build_run_config(
            {"recursion_limit": 5, "identity": {"x": 1}}, "thr", identity
        )
        self.assertEqual(run_config["configurable"]["thread_id"], "thr")
        self.assertEqual(run_config["configurable"]["user_id"], "u1")
        self.assertEqual(run_config["configurable"]["tenant_id"], "t1")
        self.assertEqual(run_config["configurable"]["roles"], ["admin"])
        self.assertEqual(run_config["recursion_limit"], 5)
        self.assertNotIn("identity", run_config)

    def test_extract_response_variants(self):
        class Msg:
            def __init__(self, content):
                self.content = content

        self.assertEqual(self.app._extract_response({"messages": [Msg("a")]}), "a")
        self.assertEqual(self.app._extract_response({"messages": [("x", "b")]}), "b")
        self.assertEqual(self.app._extract_response("plain"), "plain")
        self.assertEqual(
            self.app._extract_response({"messages": []}), "No response from agent"
        )

    def test_store_history_bounds_and_evicts(self):
        self.app.chat_histories = {}
        self.app.max_history_messages = 4
        self.app.max_threads = 2
        self.app._store_history("a", [("u", "x")] * 10)
        self.assertEqual(len(self.app.chat_histories["a"]), 4)
        self.app._store_history("b", [("u", "x")])
        self.app._store_history("c", [("u", "x")])  # exceeds cap -> evict oldest
        self.assertNotIn("a", self.app.chat_histories)
        self.assertEqual(set(self.app.chat_histories), {"b", "c"})


class AdminEndpointTests(unittest.TestCase):
    """Security regression tests for the admin-gated endpoints."""

    def setUp(self):
        self.app = _make_app(_FakeAgent())
        self.client = self.app.app.test_client()

    def tearDown(self):
        os.environ.pop("MAAP_ADMIN_TOKEN", None)

    def test_threads_requires_admin_token(self):
        # No token configured -> endpoint behaves as if it does not exist.
        resp = self.client.get("/threads")
        self.assertEqual(resp.status_code, 404)

    def test_threads_rejects_wrong_token(self):
        os.environ["MAAP_ADMIN_TOKEN"] = "correct-token"
        resp = self.client.get("/threads", headers={"X-Admin-Token": "wrong"})
        self.assertEqual(resp.status_code, 404)

    def test_threads_allows_valid_token(self):
        os.environ["MAAP_ADMIN_TOKEN"] = "correct-token"
        self.app.chat_histories["t1"] = []
        resp = self.client.get("/threads", headers={"X-Admin-Token": "correct-token"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["threads"], ["t1"])

    def test_global_reset_requires_admin_token(self):
        self.app.chat_histories["t1"] = [("user", "hi")]
        resp = self.client.post(
            "/reset", json={}, headers={"X-Requested-With": "XMLHttpRequest"}
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("t1", self.app.chat_histories)

    def test_global_reset_with_admin_token(self):
        os.environ["MAAP_ADMIN_TOKEN"] = "correct-token"
        self.app.chat_histories["t1"] = [("user", "hi")]
        resp = self.client.post(
            "/reset",
            json={},
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "X-Admin-Token": "correct-token",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.app.chat_histories, {})

    def test_per_thread_reset_does_not_require_admin_token(self):
        self.app.chat_histories["t1"] = [("user", "hi")]
        resp = self.client.post(
            "/reset",
            json={"thread_id": "t1"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("t1", self.app.chat_histories)

    def test_per_thread_reset_clears_state_provider(self):
        class _FakeStateProvider:
            def __init__(self):
                self.saved = None
                self.doc = {"tenant_id": "default", "state": {"history": [("u", "x")]}}

            def load_thread(self, thread_id):
                return self.doc

            def save_thread(self, thread_id, state, tenant_id="default", user_id="anonymous"):
                self.saved = (thread_id, state, tenant_id)
                return state

        provider = _FakeStateProvider()
        self.app.state_provider = provider
        resp = self.client.post(
            "/reset",
            json={"thread_id": "t1"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(provider.saved, ("t1", {"history": []}, "default"))

    def test_per_thread_reset_rejects_cross_tenant(self):
        class _FakeStateProvider:
            def load_thread(self, thread_id):
                return {"tenant_id": "other-tenant", "state": {}}

            def save_thread(self, *args, **kwargs):
                raise AssertionError("must not save for foreign tenant")

        self.app.state_provider = _FakeStateProvider()
        resp = self.client.post(
            "/reset",
            json={"thread_id": "t1", "identity": {"tenant_id": "default"}},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
