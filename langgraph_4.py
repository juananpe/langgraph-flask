# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key from secrets
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Check if the key was loaded successfully
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print("API keys loaded successfully.")
else:
    print("API keys not found in secrets.")

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class TweetApprovalAgent:
    def __init__(self, model, system=""):
        self.system = system
        self.model = model
        self.decision = None

        graph = StateGraph(State)

        # Add nodes
        graph.add_node("generate_post", self.generate_post)
        graph.add_node("post", self.post)
        graph.add_node("collect_feedback", self.collect_feedback)

        # Set entry point
        graph.set_entry_point("generate_post")

        # Add conditional edges - using a router function instead of a router node
        graph.add_conditional_edges(
            "generate_post",
            self.review_decision,
            {"post": "post", "feedback": "collect_feedback"}
        )

        graph.add_edge("post", END)
        graph.add_edge("collect_feedback", "generate_post")

        # Add checkpointer for interrupt functionality
        checkpointer = InMemorySaver()
        self.graph = graph.compile(checkpointer=checkpointer)

    def generate_post(self, state: State):
        messages = state["messages"]
        if self.system and len(messages) == 1:  # Only add system message on first call
            messages = [SystemMessage(content=self.system)] + messages

        result = self.model.invoke(messages)

        post_content = result.content
        print("\n📢 Current Twitter/X Post:\n")
        print(post_content)
        print("\n")

        # Use interrupt instead of input()
        response = interrupt({
            "type": "approval_request",
            "message": "Post to Twitter/X? (yes/no)",
            "post_content": post_content
        })
        
        # Handle the response
        if response and response.get("decision"):
            self.decision = response["decision"]
        else:
            self.decision = "no"  # Default to no if no response

        return {"messages": result}

    def review_decision(self, state: State) -> Literal["post", "feedback"]:
        # This is the router function that replaces the GET_REVIEW_DECISION node
        if self.decision and self.decision.lower() == "yes":
            return "post"
        else:
            return "feedback"

    def post(self, state: State):
        final_post = state["messages"][-1].content
        print("\n📢 Final Twitter/X Post:\n")
        print(final_post)
        print("\n✅ Post has been approved and is now live on Twitter/X!")
        return {"messages": state["messages"]}

    def collect_feedback(self, state: State):
        # Use interrupt instead of input()
        response = interrupt({
            "type": "feedback_request",
            "message": "How can I improve this post?",
            "current_post": state["messages"][-1].content
        })
        
        feedback = response.get("feedback", "Please make it more engaging") if response else "Please make it more engaging"
        return {"messages": [HumanMessage(content=feedback)]}

# Create the agent
llm = ChatOpenAI(model="gpt-4o")
system_prompt = "You are a professional social media manager who writes engaging Twitter/X posts."
agent = TweetApprovalAgent(llm, system=system_prompt)

# Configuration for thread management
config = {
    "configurable": {
        "thread_id": "tweet_approval_session_1"
    }
}

# Example of how to run with interrupts
print("=== Running Tweet Approval Agent with Interrupts ===")
print("Note: This will pause at interrupt points and wait for human input")
print("In a real application, you would handle these interrupts in your UI/API\n")

try:
    # Start the agent
    for chunk in agent.graph.stream(
        {"messages": [HumanMessage(content="Write a Twitter/X post on AI Agents and LangGraph")]},
        config
    ):
        print("Chunk:", chunk)
        print()
        
    print("\n=== Agent execution paused at interrupt ===")
    print("To resume, you would call:")
    print("agent.graph.invoke(Command(resume={'decision': 'yes'}), config)")
    print("or")
    print("agent.graph.invoke(Command(resume={'feedback': 'Make it more technical'}), config)")
    
except Exception as e:
    print(f"Execution stopped at interrupt: {e}")
    print("This is expected behavior - the agent is waiting for human input")