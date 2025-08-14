import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


# Load Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if API key exists
if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY not found in environment variables!")
    print("Please create a .env file with: GEMINI_API_KEY=your_api_key_here")
    exit(1)

# Gemini 2.0 Flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=gemini_api_key
)

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    try:
        response = llm.invoke(state["messages"])
        
        # Create a new state with the response appended
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [AIMessage(content=response.content)]
        
        print(f"\nAI: {response.content}")
        print("CURRENT STATE: ", new_state["messages"])
        
        return new_state
    except Exception as e:
        print(f"Error processing request: {e}")
        return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

print("Chat with AI (type 'exit' to quit):")
user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

# Save conversation to file
try:
    with open("logging.txt", "w", encoding="utf-8") as file:
        file.write("Your Conversation Log:\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n\n")
        file.write("End of Conversation")
    print("Conversation saved to logging.txt")
except Exception as e:
    print(f"Error saving conversation: {e}")
