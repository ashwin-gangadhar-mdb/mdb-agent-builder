from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import Field

from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


# Custom agent state with iteration tracking
class CustomAgentState(AgentState):
    input: str
    final_response: Optional[str] = None
    itr: int = Field(default=0)
    max_iterations: int = Field(default=3)


def create_basic_reflection_agent(
    model,
    generate_prompt: str,
    reflection_prompt: str,
    tools: Optional[List[Any]] = None,
    checkpointer: Optional[Any] = None,
    name: str = "basic_reflection_agent",
):
    """
    Create a basic reflection agent using a generate-reflect loop.

    Args:
        model: The language model to use
        generate_prompt: The prompt for the generation step
        reflection_prompt: The prompt for the reflection step
        tools: Optional list of tools to use
        checkpointer: Optional checkpointer for saving state (defaults to an
            in-memory saver created per agent)
        name: Name of the agent for logging purposes
    """
    logger.info(f"Creating basic reflection agent: {name}")

    # validate all inputs and throw error if any are missing
    if not model:
        raise ValueError("Language model is required.")
    if not generate_prompt:
        raise ValueError("Generate prompt is required.")
    if not reflection_prompt:
        raise ValueError("Reflection prompt is required.")
    logger.info(
        f"Agent {name}: Initializing with model, generate prompt, and reflection prompt"
    )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    # Build the generation sub-agent and the reflection chain ONCE, rather than
    # reconstructing them on every iteration of the loop.  The generation agent
    # is intentionally created without a checkpointer: it is an inner call made
    # without a thread_id, and the outer graph's checkpointer handles
    # persistence.
    generate_input_prompt = ChatPromptTemplate(
        [("system", generate_prompt), MessagesPlaceholder(variable_name="messages")]
    )
    generate_agent = create_react_agent(
        model=model,
        tools=tools or [],
        prompt=generate_input_prompt,
    )

    reflection_input_prompt = ChatPromptTemplate(
        [("system", reflection_prompt), MessagesPlaceholder(variable_name="messages")]
    )
    reflection_chain = reflection_input_prompt | model | StrOutputParser()

    # Define the generation function
    def generate(state: CustomAgentState, config: RunnableConfig) -> Dict[str, Any]:
        logger.info(f"Agent {name}: Generation step, iteration {state.get('itr', 0)}")
        if "input" not in state:
            logger.error(f"Agent {name}: State must contain 'input' key")
            raise ValueError("State must contain 'input' key.")
        input_text = state["input"]
        iteration = state.get("itr", 0)
        messages = state.get("messages", [])

        agent_response = generate_agent.invoke(
            {"messages": messages + [("user", f"Input: {input_text}")]}
        )

        last_message = (
            agent_response["messages"][-1].content if agent_response["messages"] else ""
        )
        # Append a single, well-typed transcript message; the ``add_messages``
        # reducer accepts ("role", "content") tuples but not bare strings.
        return {
            "messages": [("assistant", f"generate_{iteration}:\t{last_message}")],
            "final_response": last_message,
            "itr": iteration + 1,
        }

    # Define the reflection function
    def reflect(state: CustomAgentState) -> Dict[str, Any]:
        iteration = state.get("itr", 0)
        logger.info(f"Agent {name}: Reflection step, iteration {iteration}")
        reflection_response = reflection_chain.invoke(
            {"messages": state.get("messages", [])}
        )
        return {
            "messages": [
                ("assistant", f"reflection_{iteration}:\treflection: {reflection_response}")
            ]
        }

    # Define the conditional edge function
    def should_continue(state: CustomAgentState) -> str:
        iteration = state.get("itr", 0)
        logger.info(f"Agent {name}: Checking if should continue --- Iteration: {iteration}")
        logger.info(
            f"Agent {name}: ---Generation Response: {state.get('final_response')} ---"
        )
        if iteration < state.get("max_iterations", 3):
            return "continue"
        return "end"

    # Build the graph
    builder = StateGraph(CustomAgentState)
    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)
    builder.set_entry_point("generate")
    builder.add_conditional_edges(
        "generate",
        should_continue,
        {"continue": "reflect", "end": END},
    )
    builder.add_edge("reflect", "generate")

    # Compile the graph
    return builder.compile(checkpointer=checkpointer)
