from datetime import datetime
from dotenv import load_dotenv
import os
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI


REDIS_URL = "redis://localhost:6379/0"

SYSTEM_TEMPLATE_INTERVIEWER = """
# You are a member of a Design Sprint team who is tasked with interviewing an expert. Your job is to ask insightful questions of the expert, listen to their response, and then followup with another question every time they respond. However, you will only be able to ask 5 questions throughout the whole interview, and you must ask them one at a time. Tailor your questions to the expert's background so that you elicit as much interesting information from them as possible, and use interesting techniques to understand their current pain points. Never ask them questions about industries or experiences that seem like they are out of the expert's area of familiarity.

Here is the goal of this design sprint, delimited below by triple backticks. You want to get information from th expert regarding this topic:

```
{design_sprint_goal}
```

And here is some background on the person you're talking to. Make sure you ask them interesting questions based upon their background and the information you learn from their responses:

```
{expert_description}
```

Now go ahead and begin the interview process. Remember, you are holding a dialogue with a real person. That means you should only ask one question at a time and wait for them to answer before you follow up with another question. This means each of your responses should be a SINGLE question, not a chain of questions. DO NOT ANSWER THE HUMAN RESPONSES ON YOUR OWN. Just ask a question. The format of your responses should ONLY contain a question, no other dialogue. If you do not follow these rules, you will be disqualified from the study and you will severely disappoint the researchers.

Additionally, NEVER prepend your answer with "AI: "
"""

SYSTEM_TEMPLATE_EXPERT = """
You are adopting the following persona:

```
{expert_description}
```

You are being interviewed by a company who is exploring how to solve a problem around:

```
{design_sprint_goal}
```

Given this context, answer the questions they ask of you. You should really answer from your own deep, personal experience. Do not embellish or answer in ways that try to impress the interviewer. You should share your personal feelings according to your background, and you should not worry about pleasing the interviewer if your answers do not align with their problem solving mission. They are simply seeking honesty from you. Do not talk about industries outside your area of expertise, always come back to your persona to answer a question. 

Your answer should be about 150 words, and you should be detailed but concise.

Additionally, ABSOLUTELY NEVER prepend your answer with "AI: ", just answer the question. Otherwise, you will be disqualified from the study and you will severely disappoint the researchers.
"""

HUMAN_TEMPLATE = "{conversation_input}"


# Initialize the interviewer chain
def initialize_interviewer_chain(
    system_template: str, session_id: str, env: str = "dev"
):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ]
    )

    if env == "prod":
        # Load .env file
        load_dotenv()

        # Get OPENAI_API_KEY from .env file
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION")

        print("Initializing GPT 3.5...")
        model = ChatOpenAI(model="gpt-3.5-turbo")
    else:
        print("Initializing Mistral...")
        model = Ollama(model="mistral")

    chain = prompt | model

    history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL, ttl=600)

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="conversation_input",
        history_messages_key="history",
    )

    return chain_with_history


# Inoke a chain with a running history
def invoke_chain_with_history(
    chain: RunnableWithMessageHistory,
    session_id: str,
    design_sprint_goal: str = "Fallback sprint goal",
    expert_description: str = "Fallback expert bio",
    conversation_input: str = "Fallback conversation input",
):
    response = chain.invoke(
        {
            "design_sprint_goal": design_sprint_goal,
            "expert_description": expert_description,
            "conversation_input": conversation_input,
        },
        config={"configurable": {"session_id": session_id}},
    )

    return response


# Function the operates the conversation chain
def operate_conversation_chain(
    design_sprint_goal: str = "Default Goal",
    expert_description: str = "Journalist highly versed in this topic",
    num_cycles: int = 3,
    env: str = "dev",
):
    # Get the current date and time
    now = datetime.now()

    # Format it as a string
    timestamp = now.strftime("%Y%m%d%H%M%S%f")[:17]

    SESSION_INTERVIEWER = "session_interviewer_" + timestamp
    SESSION_EXPERT = "session_expert_" + timestamp

    CHAIN_INTERVIEWER = initialize_interviewer_chain(
        system_template=SYSTEM_TEMPLATE_INTERVIEWER,
        session_id=SESSION_INTERVIEWER,
        env=env,
    )

    CHAIN_EXPERT = initialize_interviewer_chain(
        system_template=SYSTEM_TEMPLATE_EXPERT, session_id=SESSION_EXPERT, env=env
    )

    next_question = invoke_chain_with_history(
        chain=CHAIN_INTERVIEWER,
        session_id=SESSION_INTERVIEWER,
        design_sprint_goal=design_sprint_goal,
        expert_description=expert_description,
        conversation_input="What would you like to ask me?",
    )

    print(f"Question: {next_question}\n\n")

    for i in range(num_cycles - 1):
        next_answer = invoke_chain_with_history(
            chain=CHAIN_EXPERT,
            session_id=SESSION_EXPERT,
            design_sprint_goal=design_sprint_goal,
            expert_description=expert_description,
            conversation_input=next_question,
        )

        print(f"Answer: {next_answer}\n\n")

        next_question = invoke_chain_with_history(
            chain=CHAIN_INTERVIEWER,
            session_id=SESSION_INTERVIEWER,
            design_sprint_goal=design_sprint_goal,
            expert_description=expert_description,
            conversation_input=next_answer,
        )

        print(f"Question: {next_question}\n\n")

    next_answer = invoke_chain_with_history(
        chain=CHAIN_EXPERT,
        session_id=SESSION_EXPERT,
        design_sprint_goal=design_sprint_goal,
        expert_description=expert_description,
        conversation_input=next_question,
    )

    print(f"Answer: {next_answer}\n")

    return next_answer
