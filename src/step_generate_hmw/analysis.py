from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def generate_hmw_question(answer: str):
    # Define the system message template with the answer as context
    system_template = """
    You are part of a Design Sprint working on the goal of:

    ```{design_sprint_goal}```

    You represent marketing, technology, and design.

    Your job is to observe an interview and take notes about interesting insights you observe from the interviewee. When you observe something interesting, convert it into a question that follows the "How might we..." format. 

    For every answer you review from an interview, generate 2 HMW questions from each of the perspectives: marketing, technology, and design. That should be 6 questions total.

    For example, if the interview answer was around buying online coffee, you might generate:

    ```
    Marketing: HMW realize they can buy coffee online?
    Tech: HMW make web experience a delight?
    Design: HWM use imagery to tell our story?
    ```

    Adhere to the following format for your response:
    {format_instructions}
    """

    class HMWQuestion(BaseModel):
        question: str = Field(description="A HMW question up to 10 words")
        role: str = Field(
            description="Role of the person asking the question, either marketing, technology, or design"
        )

    class HMWQuestions(BaseModel):
        questions: List[HMWQuestion] = Field(description="List of questions")

    parser = PydanticOutputParser(pydantic_object=HMWQuestions)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        template=system_template,
        input_variables=["design_sprint_goal"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    human_template = "{answer}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    model = ChatOpenAI(model="gpt-3.5-turbo-1106")

    chain = prompt | model | parser

    response = chain.invoke(
        {
            "answer": answer,
            "design_sprint_goal": "Create an AI tool that fully automates the Google one-week Design Sprint for a solo engineer",
        }
    )

    return response.questions
