import argparse
from dotenv import load_dotenv
import os

from step_generate_hmw.experts import generate_experts
from step_generate_hmw.interviews import operate_conversation_chain
from step_generate_hmw.analysis import generate_hmw_question

# Load .env file
load_dotenv()

# Get OPENAI_API_KEY from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION")


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


def initialize_sprint(sprint_goal, num_experts):
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
            answers = run_interview(
                design_sprint_goal=sprint_goal,
                expert_description=expert.description,
                num_cycles=4,
                env="prod",
            )

            print("Generating HMW questions...\n")
            for answer in answers:
                hmw_questions = generate_hmw(answer=answer)
            print("HMW questions generated. -----------------\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Accept arguments for goal, number of experts, and whether this is test or production."
    )
    parser.add_argument(
        "--sprint_goal", type=str, help="Sprint goal", default="Default goal"
    )
    parser.add_argument("--num_experts", type=int, help="Number of experts", default=2)

    args = parser.parse_args()

    initialize_sprint(args.sprint_goal, args.num_experts)
    # run_interview()
    # generate_hmw()
