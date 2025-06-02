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
from langgraph.types import interrupt, Command
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

# Start the agent
for chunk in agent.graph.stream(
    {"messages": [HumanMessage(content="Write a Twitter/X post on AI Agents and LangGraph")]},
    config
):
    print("Chunk:", chunk)
    
    # Check if we hit an interrupt
    if "__interrupt__" in chunk:
        interrupt_info = chunk["__interrupt__"][0]
        interrupt_data = interrupt_info.value
        
        print(f"\n=== INTERRUPT: {interrupt_data['message']} ===")
        
        if interrupt_data['type'] == 'approval_request':
            # Get real human input for approval
            user_input = input("Enter 'yes' to post or 'no' for feedback: ").strip().lower()
            
            if user_input == 'yes':
                # Resume with approval
                print("\n=== Resuming with approval ===")
                for resume_chunk in agent.graph.stream(Command(resume={'decision': 'yes'}), config):
                    print("Resume chunk:", resume_chunk)
                break
            else:
                # Start feedback loop
                print("\n=== Starting feedback loop ===")
                current_config = config
                
                while True:
                    # Resume with feedback request
                    feedback_found = False
                    for resume_chunk in agent.graph.stream(Command(resume={'decision': 'no'}), current_config):
                        print("Resume chunk:", resume_chunk)
                        
                        # Check for feedback interrupt
                        if "__interrupt__" in resume_chunk:
                            feedback_interrupt = resume_chunk["__interrupt__"][0]
                            feedback_data = feedback_interrupt.value
                            
                            if feedback_data['type'] == 'feedback_request':
                                print(f"\n=== FEEDBACK REQUEST: {feedback_data['message']} ===")
                                print(f"Current post: {feedback_data['current_post']}")
                                
                                # Get real human feedback
                                feedback = input("Enter your feedback: ").strip()
                                
                                # Resume with feedback and continue the loop
                                print(f"\n=== Applying feedback: {feedback} ===")
                                
                                # Process the feedback and wait for next approval
                                approval_found = False
                                for feedback_chunk in agent.graph.stream(Command(resume={'feedback': feedback}), current_config):
                                    print("Feedback chunk:", feedback_chunk)
                                    
                                    # Check if we get another approval request
                                    if "__interrupt__" in feedback_chunk:
                                        approval_interrupt = feedback_chunk["__interrupt__"][0]
                                        approval_data = approval_interrupt.value
                                        
                                        if approval_data['type'] == 'approval_request':
                                            print(f"\n=== NEW APPROVAL REQUEST: {approval_data['message']} ===")
                                            
                                            # Ask for approval again
                                            next_input = input("Enter 'yes' to post or 'no' for more feedback: ").strip().lower()
                                            
                                            if next_input == 'yes':
                                                # Final approval - post it
                                                print("\n=== Final approval - posting ===")
                                                for post_chunk in agent.graph.stream(Command(resume={'decision': 'yes'}), current_config):
                                                    print("Post chunk:", post_chunk)
                                                approval_found = True
                                                break
                                            else:
                                                # Continue feedback loop
                                                print("\n=== Continuing feedback loop ===")
                                                break
                                
                                if approval_found:
                                    feedback_found = True
                                    break
                                    
                    if feedback_found:
                        break
                break
        
print("\n=== Execution completed ===")