# MAAP Agent Builder Examples

These examples show supported agentic app patterns using YAML only. They are designed to be parsed and validated without external credentials. Running them requires the configured model, MongoDB, MCP, and LangChain dependencies.

| Example | Pattern | Key Capabilities |
| --- | --- | --- |
| `react_rag_mongodb.yaml` | ReAct RAG agent | MongoDB vector search, full-text search, checkpointing |
| `tool_call_mcp_agent.yaml` | Tool-calling agent | MCP tool loading over stdio |
| `reflection_quality_reviewer.yaml` | Reflection agent | Generate-reflect loop for answer improvement |
| `plan_execute_replan_research.yaml` | Planner/executor/replanner | Multi-step planning with MongoDB retrieval tools |
| `long_term_memory_assistant.yaml` | Long-term memory assistant | MongoDB Atlas vector-backed user memory |
| `governed_enterprise_support.yaml` | Governed enterprise assistant | MongoDB policies, guardrails, audit, session state |

Run an example with:

```bash
agent-builder serve --config examples/react_rag_mongodb.yaml
```
