"""Plan-Execute-Replan agent.

Generates a step-by-step plan, executes the next step with a ReAct sub-agent,
then replans based on what has been done so far until a final response is ready.
"""

import operator
from typing import Annotated, Any, List, Literal, Optional, Tuple, Type, Union

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


def get_llm_with_structured_output(llm, prompt, schema: Type[BaseModel]):
    chain = prompt | llm.with_structured_output(schema)
    return chain


def create_plan_execute_replan_agent(
    model,
    execute_prompt: str,
    tools: Optional[List[Any]] = None,
    checkpointer: Optional[Any] = None,
    name: str = "plan_execute_replan_agent",
):
    """
    Create a Plan-Execute-Replan agent that generates a plan, executes it, and can replan if needed.

    Args:
        model: The language model to use for generating plans and executing steps.
        execute_prompt: The prompt template for executing steps.
        tools: Optional list of tools the agent can use.
        checkpointer: Optional checkpointer for saving state. Defaults to an
            in-memory saver created per agent (never shared between agents).
        name: Name of the agent for logging purposes.

    Returns:
        A StateGraph that represents the agent's workflow.
    """
    logger.info(f"Creating Plan-Execute-Replan agent: {name}")

    # validate all inputs and throw error if any are missing
    if not model:
        raise ValueError("Language model is required.")
    if not execute_prompt:
        raise ValueError("Execute prompt is required.")
    logger.info(f"Agent {name}: Initializing with model and execute prompt")

    # Create a fresh checkpointer per agent so state is never shared between
    # independently created agents (a shared default instance would leak state).
    if checkpointer is None:
        checkpointer = InMemorySaver()

    if tools is None:
        tools = []
        logger.debug(f"No tools specified for agent {name}, using empty list")
    else:
        logger.debug(f"Agent {name} initialized with {len(tools)} tools")

    execute_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=execute_prompt,
        checkpointer=checkpointer,
    )

    def execute(state: PlanExecute):
        """Execute the first step in the current plan."""
        plan = state["plan"]
        if not plan:
            logger.warning(f"Agent {name}: No steps to execute in the plan")
            return {"response": "No steps to execute in the plan."}

        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        remaining_plan = plan[1:] if len(plan) > 1 else []

        logger.info(f"Agent {name}: Executing step: {task}")

        task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""

        agent_response = execute_agent.invoke({"messages": [("user", task_formatted)]})

        logger.debug(
            f"Agent {name}: Step execution complete, {len(remaining_plan)} steps remaining"
        )

        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
            "plan": remaining_plan,  # Update plan with remaining steps
        }

    def plan(state: PlanExecute):
        """Create an initial plan based on the input."""
        input_text = state["input"]
        logger.info(
            f"Agent {name}: Creating initial plan for input: {input_text[:50]}..."
        )

        plan_chain = get_llm_with_structured_output(model, planner_prompt, Plan)
        plan_response = plan_chain.invoke({"messages": [("user", input_text)]})

        logger.info(f"Agent {name}: Created plan with {len(plan_response.steps)} steps")

        return {
            "plan": plan_response.steps,
        }

    def replan(state: PlanExecute):
        """Replan based on current execution state and history."""
        logger.info(
            f"Agent {name}: Replanning after {len(state['past_steps'])} completed steps"
        )

        replan_chain = get_llm_with_structured_output(model, replanner_prompt, Act)
        replan_response = replan_chain.invoke(state)

        if isinstance(replan_response.action, Response):
            logger.info(f"Agent {name}: Replanning complete, returning final response")
            return {
                "response": replan_response.action.response,
            }
        else:
            logger.info(
                f"Agent {name}: Replanning complete, {len(replan_response.action.steps)} steps remaining"
            )
            return {
                "plan": replan_response.action.steps,
            }

    def should_end(state: PlanExecute) -> Literal["end", "continue"]:
        """Determine if the agent workflow should end or continue."""
        if "response" in state and state["response"]:
            logger.info(f"Agent {name}: Execution complete, returning response")
            return "end"
        logger.debug(f"Agent {name}: Continuing execution")
        return "continue"

    # Build the graph
    logger.info(f"Agent {name}: Building agent workflow graph")
    builder = StateGraph(PlanExecute)
    builder.add_node("plan", plan)
    builder.add_node("execute", execute)
    builder.add_node("replan", replan)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "replan")
    builder.add_conditional_edges(
        "replan",
        should_end,
        {"end": END, "continue": "execute"},
    )

    logger.info(f"Agent {name}: Compilation complete")
    return builder.compile(checkpointer=checkpointer)
