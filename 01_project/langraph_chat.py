from langgraph.graph import StateGraph, START,END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list[str], add_messages]

def chatbot(state: State):
    print(" \n\nChatbot node reached.",state)
    return {"messages": ["Hello! How can I assist you today?"]}

def sample(state: State):
    print("\n\nSample node reached.", state)
    return {"messages": ["This is a sample state."]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("sample", sample)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "sample")
graph_builder.add_edge("sample", END)

graph = graph_builder.compile()

updated_graph = graph.invoke(State({"messages": ["Hello!"]}))
print("\n\nupdated_graph:", updated_graph)
