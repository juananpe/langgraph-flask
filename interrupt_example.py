#!/usr/bin/env python3
"""
Example showing how to handle LangGraph interrupts for human-in-the-loop workflows.
This demonstrates the proper way to resume execution after an interrupt.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def create_tweet_agent():
    """Create a tweet approval agent with interrupt functionality."""
    
    def generate_post(state: State):
        llm = ChatOpenAI(model="gpt-4o")
        messages = state["messages"]
        
        # Add system message
        system_msg = SystemMessage(content="You are a professional social media manager who writes engaging Twitter/X posts.")
        messages = [system_msg] + messages
        
        result = llm.invoke(messages)
        post_content = result.content
        
        print(f"\n📢 Generated Post:\n{post_content}\n")
        
        # Interrupt for human approval
        response = interrupt({
            "type": "approval_request",
            "message": "Do you want to post this tweet?",
            "post_content": post_content,
            "options": ["approve", "edit", "reject"]
        })
        
        return {"messages": [result]}
    
    def handle_approval(state: State):
        """Route based on human decision."""
        # This would be set by the resume command
        return "post"  # Default action
    
    def post_tweet(state: State):
        final_post = state["messages"][-1].content
        print(f"\n✅ Tweet Posted: {final_post}")
        return {"messages": state["messages"]}
    
    def collect_feedback(state: State):
        # Interrupt for feedback
        response = interrupt({
            "type": "feedback_request", 
            "message": "How should we improve this post?",
            "current_post": state["messages"][-1].content
        })
        
        return {"messages": state["messages"]}
    
    # Build the graph
    graph = StateGraph(State)
    graph.add_node("generate", generate_post)
    graph.add_node("post", post_tweet)
    graph.add_node("feedback", collect_feedback)
    
    graph.set_entry_point("generate")
    graph.add_edge("generate", "post")  # Simplified for demo
    graph.add_edge("post", END)
    
    # Add checkpointer for interrupt functionality
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

def main():
    """Demonstrate interrupt handling."""
    agent = create_tweet_agent()
    
    config = {
        "configurable": {
            "thread_id": "demo_session"
        }
    }
    
    print("=== Starting Tweet Generation with Interrupts ===\n")
    
    try:
        # Initial invocation
        result = agent.invoke(
            {"messages": [HumanMessage(content="Write a tweet about LangGraph interrupts")]},
            config
        )
        print("Initial result:", result)
        
    except Exception as e:
        print(f"Execution paused at interrupt: {type(e).__name__}")
        print("This is expected - the agent is waiting for human input\n")
        
        # Show how to resume with approval
        print("=== Resuming with Approval ===")
        try:
            resume_result = agent.invoke(
                Command(resume={"decision": "approve"}),
                config
            )
            print("Resume result:", resume_result)
        except Exception as e2:
            print(f"Still at interrupt: {type(e2).__name__}")
            
        # Show how to get current state
        print("\n=== Current State ===")
        current_state = agent.get_state(config)
        print(f"Current node: {current_state.next}")
        print(f"Interrupt data: {current_state.tasks}")

if __name__ == "__main__":
    main()
