import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.mongodb import MongoDBSaver
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from openai import OpenAI

MONGODB_URI = os.getenv("MONGODB_URI")

client = OpenAI(api_key=os.getenv("OPENAI_API"))  # Replace with your actual OpenAI API key


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    print(" \n\nChatbot node reached.", state)

    # Convert the graph's message history into OpenAI's {role, content} format.
    # add_messages stores LangChain message objects; .type is "human"/"ai"/"system".
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    openai_messages = [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in state["messages"]
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,
    )
    reply = response.choices[0].message.content

    return {"messages": [{"role": "assistant", "content": reply}]}

def sample(state: State):
    print("\n\nSample node reached.", state)
    return {"messages": ["This is a sample state."]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("sample", sample)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "sample")
graph_builder.add_edge("sample", END)

# The checkpointer (and its Mongo connection) is only alive inside this block,
# so both compile() and invoke() must run within it.
with MongoDBSaver.from_conn_string(MONGODB_URI, db_name="langgraph") as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)

    # thread_id keys the persisted state. Reuse the same id to resume a conversation.
    config = {"configurable": {"thread_id": "aman-1"}}

    updated_graph = graph.invoke(
        {"messages": [{"role": "user", "content": "what is my name?"}]},
        config=config,
    )
    print("\n\n=== Conversation ===")
    for m in updated_graph["messages"]:
        m.pretty_print()
