import argparse
from dotenv import load_dotenv
import os
import json

from step_generate_hmw.experts import generate_experts
from step_generate_hmw.interviews import operate_conversation_chain
from step_generate_hmw.analysis import generate_hmw_question

# Load .env file
load_dotenv()

# Get keys from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


def generate_hmw(answer: str):
    questions = generate_hmw_question(answer=answer)

    for question in questions:
        print(question)
    return questions


def run_interview(
    design_sprint_goal: str, expert_description: str, num_cycles: int, env: str
):
    answers = operate_conversation_chain(
        design_sprint_goal=design_sprint_goal,
        expert_description=expert_description,
        num_cycles=num_cycles,
        env=env,
    )

    return answers


def initialize_sprint():
    sprint_goal = None
    num_experts = None

    if not sprint_goal:
        while not sprint_goal:
            sprint_goal = input("Enter the sprint goal: ")
            if sprint_goal == "":
                print("Sprint goal is required. Please enter a sprint goal.")

    if not num_experts:
        while not num_experts:
            num_experts = input("Enter the number of experts: ")
            if num_experts == "":
                print("Number of experts is required. Please enter a number.")
            else:
                num_experts = int(num_experts)

    print("Running a new design sprint\n")
    print("Sprint goal: " + sprint_goal + "\n")

    # Generate the experts
    print("Generating experts...\n")
    experts = generate_experts(sprint_goal, num_experts)
    print("Experts generated. -----------------\n")

    # For each expert, run the interview
    print("Interviewing experts...\n")
    if not experts:
        print("No experts found.")
        return
    else:
        for expert in experts:
            print(f"Interviewing expert: {expert.name} - {expert.description}...\n")
            conversation_log = run_interview(
                design_sprint_goal=sprint_goal,
                expert_description=expert.description,
                num_cycles=2,
                env="prod",
            )

            answers = [
                item["message"] for item in conversation_log if item["type"] == "answer"
            ]

            print("Generating HMW questions...\n")
            for answer in answers:
                hmw_questions = generate_hmw(
                    answer=answer, design_sprint_goal=sprint_goal
                )
            print("HMW questions generated. -----------------\n")

    return


# For local
if __name__ == "__main__":
    initialize_sprint()
