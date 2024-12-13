import os
from dotenv import load_dotenv
from typing import List, Annotated, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from IPython.display import Image, display
import uuid
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from then:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requiremehts that the output MUST adhere to

If you are not able to discern this info, ask them to clarity! Do not attempt to widly guess.

After you are able to discern all the information, call the relevant tool.
"""

def get_messages_info(message): 
    return [SystemMessage(content=template)] + message

class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])

def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

prompt_system ="""Based on the folling requirements, write a good and professional prompt template that can be used to generate prompts for the user. {reqs}"""

def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
            print(tool_call)
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)

    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs

def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

def get_state(state) -> Literal["prompt", "info", "__end__"]:
    message = state["messages"]
    if isinstance(message[-1], AIMessage) and message[-1].tool_calls:
        return "prompt"
    elif not isinstance(message[-1], HumanMessage):
        return END
    return 'info'

class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()
workflow = StateGraph(State)

workflow.add_node('info', info_chain)
workflow.add_node('prompt', prompt_gen_chain)

workflow.add_conditional_edges('info', get_state)
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")

graph = workflow.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": { "thread_id": str(uuid.uuid4()) }}

while True:
    user = input("User (q/Q to quit): ")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break
    output = None
    for output in graph.stream({ "messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()
    if output and "prompt" in output:
        print("AI: Here is the prompt: ", output["prompt"])