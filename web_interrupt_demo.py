#!/usr/bin/env python3
"""
Flask web app demonstrating LangGraph interrupts in a web interface.
This shows how to handle human-in-the-loop workflows in a real application.
"""

from flask import Flask, render_template_string, request, jsonify, session
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Global agent instance
agent = None
checkpointer = InMemorySaver()

def create_tweet_agent():
    """Create a tweet approval agent with interrupt functionality."""
    
    def generate_post(state: State):
        llm = ChatOpenAI(model="gpt-4o")
        messages = state["messages"]
        
        # Add system message if this is the first call
        if len(messages) == 1:
            system_msg = SystemMessage(content="You are a professional social media manager who writes engaging Twitter/X posts. Create compelling content that drives engagement.")
            messages = [system_msg] + messages
        
        result = llm.invoke(messages)
        
        print(f"\n📢 Generated Post:\n{result.content}\n")
        
        # Interrupt for human approval BEFORE returning
        interrupt({
            "type": "approval_request",
            "message": "Do you want to post this tweet?",
            "post_content": result.content,
            "options": ["approve", "edit", "reject"]
        })
        
        return {"messages": [result]}
    
    def route_decision(state: State):
        """Route based on human decision from interrupt resume."""
        # This will be populated by Command(resume=...) 
        return "post"  # Default - will be overridden by resume data
    
    def post_tweet(state: State):
        final_post = state["messages"][-1].content
        print(f"\n✅ Tweet Posted: {final_post}")
        return {"messages": state["messages"]}
    
    def collect_feedback(state: State):
        # This node will receive feedback via Command(resume={"feedback": "..."})
        # No interrupt needed here - just process the feedback and move to regenerate
        return {"messages": state["messages"]}
    
    def regenerate_with_feedback(state: State):
        """Regenerate the post with feedback."""
        llm = ChatOpenAI(model="gpt-4o")
        
        # The feedback should be in the last message added by the resume command
        original_post = state["messages"][-2].content  # Get the AI-generated post
        feedback = state["messages"][-1].content  # Get the human feedback
        
        # Create a prompt that includes both the original post and feedback
        improvement_prompt = f"""
        Original post: {original_post}
        
        User feedback: {feedback}
        
        Please create an improved version of the post based on the user's feedback.
        """
        
        # Add system message and improvement prompt
        system_msg = SystemMessage(content="You are a professional social media manager who writes engaging Twitter/X posts. Improve posts based on user feedback.")
        messages = [system_msg, HumanMessage(content=improvement_prompt)]
        
        result = llm.invoke(messages)
        
        # Interrupt again for approval of the improved post BEFORE returning
        interrupt({
            "type": "approval_request",
            "message": "Review the improved post",
            "post_content": result.content,
            "options": ["approve", "edit", "reject"]
        })
        
        return {"messages": state["messages"] + [result]}
    
    # Build the graph
    graph = StateGraph(State)
    graph.add_node("generate", generate_post)
    graph.add_node("post", post_tweet)
    graph.add_node("feedback", collect_feedback)
    graph.add_node("regenerate", regenerate_with_feedback)
    
    graph.set_entry_point("generate")
    
    # Add conditional routing from generate
    # When interrupt happens, execution pauses and resume determines the next node
    graph.add_conditional_edges(
        "generate",
        route_decision,
        {"post": "post", "feedback": "feedback"}
    )
    
    # Add edge from feedback to regenerate
    graph.add_edge("feedback", "regenerate")
    
    # Add conditional routing from regenerate
    # After regeneration, interrupt again for approval
    graph.add_conditional_edges(
        "regenerate", 
        route_decision,
        {"post": "post", "feedback": "feedback"}
    )
    
    graph.add_edge("post", END)
    
    # Add checkpointer for interrupt functionality
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Interrupt Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .post-content { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .buttons { margin: 10px 0; }
        button { margin: 5px; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
        .approve { background: #4CAF50; color: white; }
        .reject { background: #f44336; color: white; }
        .edit { background: #2196F3; color: white; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        .status { padding: 10px; margin: 10px 0; border-radius: 3px; }
        .success { background: #d4edda; color: #155724; }
        .info { background: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <h1>🐦 Tweet Approval with LangGraph Interrupts</h1>
    
    <div id="status"></div>
    
    <form id="topicForm">
        <label>Enter tweet topic:</label><br>
        <input type="text" id="topic" placeholder="e.g., AI agents and automation" style="width: 300px; padding: 5px;">
        <button type="submit">Generate Tweet</button>
    </form>
    
    <div id="tweetContent" style="display: none;">
        <h3>Generated Tweet:</h3>
        <div id="postContent" class="post-content"></div>
        
        <div class="buttons">
            <button class="approve" onclick="handleDecision('approve')">✅ Approve & Post</button>
            <button class="edit" onclick="showFeedbackForm()">✏️ Request Changes</button>
            <button class="reject" onclick="handleDecision('reject')">❌ Reject</button>
        </div>
    </div>
    
    <div id="feedbackForm" style="display: none;">
        <h3>Provide Feedback:</h3>
        <textarea id="feedback" placeholder="How should we improve this tweet?"></textarea>
        <button onclick="submitFeedback()">Submit Feedback</button>
    </div>

    <script>
        let currentThreadId = null;
        
        document.getElementById('topicForm').onsubmit = async function(e) {
            e.preventDefault();
            const topic = document.getElementById('topic').value;
            if (!topic) return;
            
            showStatus('Generating tweet...', 'info');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({topic: topic})
                });
                
                const data = await response.json();
                
                if (data.status === 'interrupted') {
                    currentThreadId = data.thread_id;
                    showTweet(data.post_content);
                    showStatus('Tweet generated! Please review and decide.', 'info');
                } else {
                    showStatus('Error: ' + data.message, 'error');
                }
            } catch (error) {
                showStatus('Error generating tweet: ' + error.message, 'error');
            }
        };
        
        function showTweet(content) {
            document.getElementById('postContent').textContent = content;
            document.getElementById('tweetContent').style.display = 'block';
            document.getElementById('feedbackForm').style.display = 'none';
        }
        
        function showFeedbackForm() {
            document.getElementById('feedbackForm').style.display = 'block';
        }
        
        async function handleDecision(decision) {
            if (!currentThreadId) return;
            
            showStatus('Processing decision...', 'info');
            
            try {
                const response = await fetch('/resume', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        thread_id: currentThreadId,
                        decision: decision
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'completed') {
                    showStatus(data.message, 'success');
                    document.getElementById('tweetContent').style.display = 'none';
                    currentThreadId = null;
                } else if (data.status === 'feedback_requested') {
                    showStatus(data.message, 'info');
                    showFeedbackForm();
                } else {
                    showStatus('Error: ' + data.message, 'error');
                }
                
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }
        
        async function submitFeedback() {
            const feedback = document.getElementById('feedback').value;
            if (!feedback || !currentThreadId) return;
            
            showStatus('Submitting feedback...', 'info');
            
            try {
                const response = await fetch('/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        thread_id: currentThreadId,
                        feedback: feedback
                    })
                });
                
                const data = await response.json();
                if (data.status === 'regenerated') {
                    showTweet(data.post_content);
                    showStatus(data.message, 'info');
                } else {
                    showStatus(data.message, 'info');
                }
                
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + type;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_tweet():
    global agent
    if not agent:
        agent = create_tweet_agent()
    
    data = request.json
    topic = data.get('topic', '')
    
    # Create unique thread ID
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Start the agent - this will hit the interrupt
        agent.invoke(
            {"messages": [HumanMessage(content=f"Write a Twitter/X post about: {topic}")]},
            config
        )
    except Exception:
        # Expected - agent hit interrupt
        pass
    
    # Get the current state to extract the interrupt data
    state = agent.get_state(config)
    
    if state.tasks:
        # Extract the post content from the interrupt
        interrupt_obj = state.tasks[0].interrupts[0]
        # The interrupt value is stored in the 'value' attribute
        interrupt_data = interrupt_obj.value
        post_content = interrupt_data.get('post_content', 'No content generated')
        
        return jsonify({
            'status': 'interrupted',
            'thread_id': thread_id,
            'post_content': post_content
        })
    
    return jsonify({'status': 'error', 'message': 'No interrupt found'})

@app.route('/resume', methods=['POST'])
def resume_execution():
    global agent
    data = request.json
    thread_id = data.get('thread_id')
    decision = data.get('decision')
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        if decision == 'approve':
            # Resume and go to post node
            result = agent.invoke(Command(resume="post"), config)
            return jsonify({
                'status': 'completed',
                'decision': 'approved',
                'message': '✅ Tweet approved and posted!'
            })
        elif decision == 'reject':
            # Just end the process
            return jsonify({
                'status': 'completed', 
                'decision': 'rejected',
                'message': '❌ Tweet rejected'
            })
        else:
            # For any other decision, go to feedback
            result = agent.invoke(Command(resume="feedback"), config)
            return jsonify({
                'status': 'feedback_requested',
                'message': 'Please provide feedback to improve the tweet'
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    global agent
    data = request.json
    thread_id = data.get('thread_id')
    feedback = data.get('feedback')
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Add the feedback as a human message and resume to feedback node
        from langchain_core.messages import HumanMessage
        
        # First, add the feedback message to the state
        agent.update_state(
            config,
            {"messages": [HumanMessage(content=feedback)]}
        )
        
        # Then resume to the regenerate node (feedback node will pass through to regenerate)
        agent.invoke(Command(resume="feedback"), config)
        
    except Exception as e:
        # Expected - agent will hit interrupt in regenerate node
        pass
    
    # Get the new state with the improved post
    state = agent.get_state(config)
    
    if state.tasks:
        # Extract the new post content from the interrupt
        interrupt_obj = state.tasks[0].interrupts[0]
        interrupt_data = interrupt_obj.value
        new_post_content = interrupt_data.get('post_content', 'No content generated')
        
        return jsonify({
            'status': 'regenerated',
            'post_content': new_post_content,
            'message': 'Tweet regenerated with your feedback!'
        })
    
    return jsonify({'status': 'error', 'message': 'Failed to regenerate tweet'})

if __name__ == '__main__':
    print("Starting LangGraph Interrupt Demo...")
    print("Visit http://localhost:4444 to see the demo")
    app.run(debug=True, port=4444)
