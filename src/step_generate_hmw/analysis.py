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


def generate_hmw_question(answer: str, design_sprint_goal: str):
    # Define the system message template with the answer as context
    system_template = """
    You are part of a Design Sprint working on the goal of:

    ```{design_sprint_goal}```

    You represent experts in product management, technology, and design.

    Your job is to observe an interview and take notes about interesting insights you observe from the interviewee. When you observe something interesting that actually could generate a unique, specific insight, convert it into a question that follows the "How might we..." format, and use shorthand HMW to denote it.

    For every answer you review from an interview, generate 1 HMW question from each of the perspectives: product, technology, and design. That should be 3 questions total.

    For example, if the interview answer was around buying online coffee, you might generate:

    ```
    Marketing: HMW help people realize they can order coffee online?
    Tech: HMW remember the user's favorite orders to make ordering simple?
    Design: HMW make the user feel like a regular on the site?
    ```

    DO NOT EVER include generic questions like "HMW ensure the web app provides a seamless user experience?", that is a terrible HMW question that provides no valuable insight. Your questions must be more specific to the interviewee's answer and provide new insight. You will severely disappoint the expert if you provide uninsightful HMW questions.

    Adhere to the following format for your response:
    {format_instructions}
    """

    class HMWQuestion(BaseModel):
        question: str = Field(
            description="A HMW question up to 10 words. This should always be in the format: HMW <question>?"
        )
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

    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    chain = prompt | model | parser

    response = chain.invoke(
        {
            "answer": answer,
            "design_sprint_goal": design_sprint_goal,
        }
    )

    return response.questions
