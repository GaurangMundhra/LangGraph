from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
import os


load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

def process(state : AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

user_input = input("Enter: ")

while user_input != "exit":
    agent.invoke({"messages" : [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
