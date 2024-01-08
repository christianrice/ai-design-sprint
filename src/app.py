import argparse

from step_generate_hmw.experts import generate_experts
from step_generate_hmw.interviews import operate_conversation_chain


def initialize_sprint(sprint_goal, num_experts):
    experts = generate_experts(sprint_goal, num_experts)

    if not experts.experts:
        print("No experts found.")
    else:
        for expert in experts.experts:
            print(expert.name + ": " + expert.description)


def run_interview():
    operate_conversation_chain(
        design_sprint_goal="Create an AI tool that fully automates the Google one-week Design Sprint for a solo engineer",
        expert_description="Design Sprint Facilitator who has run hundreds of design sprints in-person with emerging technology teams",
        num_cycles=2,
        env="prod",
    )

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

    # initialize_sprint(args.sprint_goal, args.num_experts)
    run_interview()
