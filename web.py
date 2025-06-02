# -*- coding: utf-8 -*-
import os
import uuid
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Load API key from secrets
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Check if the key was loaded successfully
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print("API keys loaded successfully.")
else:
    print("API keys not found in secrets.")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

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
        return {"messages": result}

    def get_approval(self, state: State):
        # Use interrupt to get user approval
        decision = interrupt("Post to Twitter/X? (yes/no): ")
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
        return {"messages": state["messages"]}

    def collect_feedback(self, state: State):
        # Use interrupt to get user feedback
        feedback = interrupt("How can I improve this post? ")
        return {"messages": [HumanMessage(content=feedback)], "user_feedback": feedback}

# Global agent instance
llm = ChatOpenAI(model="gpt-4o")
system_prompt = "You are a professional social media manager who writes engaging Twitter/X posts."
agent = TweetApprovalAgent(llm, system=system_prompt)

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/start_generation', methods=['POST'])
def start_generation():
    try:
        data = request.get_json()
        topic = data.get('topic', 'AI Agents and LangGraph')
        
        # Create a new thread ID for this session
        thread_id = str(uuid.uuid4())
        session['thread_id'] = thread_id
        session['status'] = 'generating'
        
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        # Start the generation process
        interrupt_prompt = None
        generated_post = None
        
        for chunk in agent.graph.stream({
            "messages": [HumanMessage(content=f"Write a Twitter/X post on {topic}")]
        }, config):
            if "__interrupt__" in chunk:
                interrupt_prompt = chunk["__interrupt__"][0].value
                session['interrupt_prompt'] = interrupt_prompt
                session['status'] = 'waiting_approval'
            elif "generate_post" in chunk and chunk["generate_post"]:
                # Extract the generated post
                messages = chunk["generate_post"]["messages"]
                if messages:
                    generated_post = messages.content
                    session['generated_post'] = generated_post
        
        return jsonify({
            'success': True,
            'thread_id': thread_id,
            'generated_post': generated_post,
            'interrupt_prompt': interrupt_prompt,
            'status': session['status']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/respond_to_interrupt', methods=['POST'])
def respond_to_interrupt():
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        thread_id = session.get('thread_id')
        
        if not thread_id:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        # Resume with user input
        completed = False
        interrupt_prompt = None
        final_post = None
        new_generated_post = None
        
        for chunk in agent.graph.stream(Command(resume=user_input), config):
            if "post" in chunk and chunk["post"]:
                completed = True
                final_post = chunk["post"]["messages"][-1].content
                session['final_post'] = final_post
                session['status'] = 'completed'
                break
            elif "__interrupt__" in chunk:
                interrupt_prompt = chunk["__interrupt__"][0].value
                session['interrupt_prompt'] = interrupt_prompt
                if "How can I improve" in interrupt_prompt:
                    session['status'] = 'waiting_feedback'
                else:
                    session['status'] = 'waiting_approval'
            elif "generate_post" in chunk and chunk["generate_post"]:
                # New post was generated after feedback
                messages = chunk["generate_post"]["messages"]
                if messages:
                    new_generated_post = messages.content
                    session['generated_post'] = new_generated_post
        
        response = {
            'success': True,
            'completed': completed,
            'interrupt_prompt': interrupt_prompt,
            'status': session['status']
        }
        
        if completed:
            response['final_post'] = final_post
        elif new_generated_post:
            response['generated_post'] = new_generated_post
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_status')
def get_status():
    return jsonify({
        'thread_id': session.get('thread_id'),
        'status': session.get('status'),
        'generated_post': session.get('generated_post'),
        'final_post': session.get('final_post'),
        'interrupt_prompt': session.get('interrupt_prompt')
    })

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
