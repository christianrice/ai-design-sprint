import os
from flask import Flask, request, jsonify
from step_generate_hmw.experts import generate_experts
from step_generate_hmw.interviews import operate_conversation_chain
from step_generate_hmw.analysis import generate_hmw_question

app = Flask(__name__)

# Get keys from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


@app.route("/generate_experts", methods=["POST"])
def generate_experts_route():
    data = request.get_json()
    design_sprint_goal = data.get("design_sprint_goal")
    num_experts = data.get("num_experts")

    experts = generate_experts(sprint_goal=design_sprint_goal, num_experts=num_experts)
    return jsonify([expert.dict() for expert in experts])


@app.route("/conduct_interview", methods=["POST"])
def conduct_interview_route():
    data = request.get_json()
    design_sprint_goal = data.get("design_sprint_goal")
    expert_description = data.get("expert_description")
    num_cycles = data.get("num_cycles", 3)
    env = data.get("env", "dev")

    conversation_log = operate_conversation_chain(
        design_sprint_goal=design_sprint_goal,
        expert_description=expert_description,
        num_cycles=num_cycles,
        env=env,
    )

    return conversation_log


@app.route("/generate_hmw_questions", methods=["POST"])
def generate_hmw_questions_route():
    data = request.get_json()
    answer = data.get("answer")
    design_sprint_goal = data.get("design_sprint_goal")

    questions = generate_hmw_question(
        answer=answer, design_sprint_goal=design_sprint_goal
    )

    return jsonify([question.dict() for question in questions])
