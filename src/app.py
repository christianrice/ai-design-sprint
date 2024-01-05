import argparse

from generate_experts.experts import get_experts


def initialize_sprint(sprint_goal, num_experts):
    experts = get_experts(sprint_goal, num_experts)

    if not experts.experts:
        print("No experts found.")
    else:
        for expert in experts.experts:
            print(expert.name + ": " + expert.description)


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
