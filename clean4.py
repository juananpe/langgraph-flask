# -*- coding: utf-8 -*-
import os
import uuid
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

from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_decision: Optional[str]
    user_feedback: Optional[str]

class TweetApprovalAgent:
    def __init__(self, model, system=""):
        self.system = system
        self.model = model

        graph = StateGraph(State)

        # Add nodes
        graph.add_node("generate_post", self.generate_post)
        graph.add_node("get_approval", self.get_approval)
        graph.add_node("post", self.post)
        graph.add_node("collect_feedback", self.collect_feedback)

        # Set entry point
        graph.set_entry_point("generate_post")

        # Add edges
        graph.add_edge("generate_post", "get_approval")
        graph.add_conditional_edges(
            "get_approval",
            self.review_decision,
            {"post": "post", "feedback": "collect_feedback"}
        )

        graph.add_edge("post", END)
        graph.add_edge("collect_feedback", "generate_post")

        # A checkpointer must be enabled for interrupts to work!
        checkpointer = MemorySaver()
        self.graph = graph.compile(checkpointer=checkpointer)

    def generate_post(self, state: State):
        messages = state["messages"]
        if self.system and len(messages) == 1:  # Only add system message on first call
            messages = [SystemMessage(content=self.system)] + messages

        result = self.model.invoke(messages)

        # Show the generated post
        post_content = result.content
        print("\n📢 Current Twitter/X Post:\n")
        print(post_content)
        print("\n")

        return {"messages": result}

    def get_approval(self, state: State):
        # Use interrupt to get user approval
        decision = interrupt("Post to Twitter/X? (yes/no): ")
        print(f"> Received decision: {decision}")
        return {"user_decision": decision}

    def review_decision(self, state: State) -> Literal["post", "feedback"]:
        # Check the user's decision
        decision = state.get("user_decision", "").lower()
        if decision == "yes":
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
        # Use interrupt to get user feedback
        feedback = interrupt("How can I improve this post? ")
        print(f"> Received feedback: {feedback}")
        return {"messages": [HumanMessage(content=feedback)], "user_feedback": feedback}

# Create the agent
llm = ChatOpenAI(model="gpt-4o")
system_prompt = "You are a professional social media manager who writes engaging Twitter/X posts."
agent = TweetApprovalAgent(llm, system=system_prompt)

# Configuration for the graph execution
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

# Start the tweet generation process
print("🚀 Starting tweet generation process...")

# Initial execution
interrupt_prompt = None
for chunk in agent.graph.stream({
    "messages": [HumanMessage(content="Write a Twitter/X post on AI Agents and LangGraph")]
}, config):
    # print(f"Chunk: {chunk}")
    # Check if this chunk contains an interrupt
    if "__interrupt__" in chunk:
        interrupt_prompt = chunk["__interrupt__"][0].value
        print(f"\n❓ {interrupt_prompt}", end="")

# Continue handling interrupts until the process is complete
while True:
    try:
        # Get user input for the current interrupt
        user_input = input()
        print(f"\n📋 Resuming with: {user_input}")
        
        # Resume with user input
        completed = False
        interrupt_prompt = None
        for chunk in agent.graph.stream(Command(resume=user_input), config):
            # print(f"Chunk: {chunk}")
            # Check if we've reached the end (post node completed)
            if "post" in chunk and chunk["post"]:
                completed = True
                break
            # Check if this chunk contains an interrupt
            elif "__interrupt__" in chunk:
                interrupt_prompt = chunk["__interrupt__"][0].value
                print(f"\n❓ {interrupt_prompt}", end="")
        
        # If the post was completed, break out of the loop
        if completed:
            print("\n🎉 Process completed!")
            break
            
    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user")
        break
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        break