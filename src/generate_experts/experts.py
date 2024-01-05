from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


def get_experts(sprint_goal: str = "Default goal", num_experts: int = 1):
    class Expert(BaseModel):
        name: str = Field(description="Name of the expert")
        description: str = Field(
            description="Description of the expert in 20 words or less"
        )

    class Experts(BaseModel):
        experts: List[Expert] = Field(description="List of experts")

    output_parser = PydanticOutputParser(pydantic_object=Experts)

    prompt_template = """
    You are a member of a Design Sprint who is tasked with finding a panel of experts on the following design sprint goal: 

    ```
    {sprint_goal}
    ```

    Define {num_experts} different dream personas of experts who could help with this scenario.

    For example, if the sprint problem were "Bring great coffee to new customers online" you would provide {num_experts} personas similar to:

    ```
    Steve
    Casual coffee drinker who sometimes goes to Starbucks but usually makes Folgers at home.

    Brian
    Coffee snob who roasts coffee at home, hand grinds it, and perfectly measures to the gram his morning cup of coffee.
    ```

    {format_instructions}

    Respond with NOTHING else but the valid JSON described above. Do not return a list. Do not return any preamble. Just return the JSON and nothing else at all.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["field_of_expertise"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    model = Ollama(model="mistral")

    chain = prompt | model | output_parser

    try:
        response = chain.invoke(
            {"sprint_goal": sprint_goal, "num_experts": num_experts}
        )
    except Exception as e:
        print(f"Error parsing output: {e}")
        response = Experts(experts=[])

    return response
