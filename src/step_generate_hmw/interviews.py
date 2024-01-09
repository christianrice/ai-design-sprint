from datetime import datetime
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
from langchain.schema.output_parser import StrOutputParser


REDIS_URL = "redis://localhost:6379/0"

SYSTEM_TEMPLATE_INTERVIEWER = """
# You are a member of a Design Sprint team who is tasked with interviewing an expert. Your job is to ask insightful questions of the expert, listen to their response, and then follow up with an insightful new question every time they respond. However, you will only be able to ask a total of 5 questions throughout the whole interview, and you can only ask them one at a time. Tailor your questions to the expert's background so that you elicit as much valuable information from them as possible, and make an effort to understand their perspective so you can ask nuanced questions. Remember, this is an expert so don't waste your questions and do not ask them about industries or experiences that are out of their area of expertise.

Here is the goal of this design sprint, delimited below by triple backticks. You want to get information from the expert regarding this topic:

```
{design_sprint_goal}
```

And here is background on the expert you're talking to. Ask them interesting questions based upon their background and from your dialogue with them:

```
{expert_description}
```

Now begin the interview process. Remember, you are holding a dialogue with a real person. You should only ask one question at a time. The format of your responses should ONLY contain a question, no other dialogue. If you do not follow these rules, you will be disqualified from the study and you will severely disappoint the researchers."
"""

SYSTEM_TEMPLATE_EXPERT = """
You are an expert with the following persona:

```
{expert_description}
```

You are being interviewed by a company who is exploring how to solve a problem around:

```
{design_sprint_goal}
```

Given this context, answer the questions they ask of you. Your answers should come from your own deep, personal experience. Do not embellish or answer in ways that try to impress the interviewer. You should share your personal feelings according to your background, and you should not worry about pleasing the interviewer if your answers do not align with their problem solving mission. They are simply seeking honesty from you. Do not talk about industries outside your area of expertise, always come back to your persona to answer a question. Furthermore, don't waste time repeating their question. Just get right into answering it succinctly and clearly, with no wasted words.

Your answer should be about 150 words, and you should be detailed but concise.
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
        model = ChatOpenAI(model="gpt-3.5-turbo-1106")
        # model = ChatOpenAI(model="gpt-4-1106-preview")
        output_parser = StrOutputParser()

        chain = prompt | model | output_parser
    else:
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

    # Initialize the log as an empty list
    answer_log = []

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

        answer_log.append(next_answer)

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

    answer_log.append(next_answer)

    print(f"Answer: {next_answer}\n")
    print("----------------------------------------\n\n")

    return answer_log
