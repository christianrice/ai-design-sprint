from uuid import uuid4
from logger import logger
from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI


class Expert(BaseModel):
    name: str = Field(description="Name of the expert")
    description: str = Field(
        description="Description of the expert in 20 words or less"
    )


class ExpertWithID(Expert):
    id: str = Field(default_factory=lambda: str(uuid4()))


class Experts(BaseModel):
    experts: List[Expert] = Field(description="List of experts")


class ExpertsWithID(BaseModel):
    experts: List[ExpertWithID] = Field(default_factory=list)


def generate_experts(sprint_goal: str = "Default goal", num_experts: int = 1):
    output_parser = PydanticOutputParser(pydantic_object=Experts)

    prompt_template = """
    You are a member of a Design Sprint who is tasked with finding a panel of experts on the following design sprint goal: 

    ```
    {sprint_goal}
    ```

    Define {num_experts} different dream personas of experts who could help with this scenario.

    For example, if the sprint problem were "Bring great coffee to new customers online" you would provide personas similar to:

    ```
    Steve
    Casual coffee drinker who sometimes goes to Starbucks but usually makes Folgers at home.

    Brian
    Coffee snob who roasts coffee at home, hand grinds it, and perfectly measures to the gram his morning cup of coffee.
    ```

    {format_instructions}

    Respond with NOTHING else but the valid JSON described above for the {num_experts} experts you have created.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["spring_goal", "num_experts"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    chain = prompt | model | output_parser

    experts_with_id = ExpertsWithID()

    try:
        response = chain.invoke(
            {"sprint_goal": sprint_goal, "num_experts": num_experts}
        )

        for expert in response.experts:
            expert_with_id = ExpertWithID(**expert.dict())
            print(
                f"Expert: {expert_with_id.name}, Description: {expert_with_id.description}"
            )
            experts_with_id.experts.append(expert_with_id)
    except Exception as e:
        logger.error(f"Error parsing output: {e}")
    return experts_with_id.experts
