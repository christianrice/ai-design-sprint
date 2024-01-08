from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama


REDIS_URL = "redis://localhost:6379/0"


# Initialize the conversation chain
def initialize_conversation_chain_a(session_id: str):
    INTERVIEW_SYSTEM_TEMPLATE = """
    # You are a member of a Design Sprint team who is tasked with interviewing an expert. Your job is to ask insightful questions of the expert, listen to their response, and then followup with another question every time they respond. However, you will only be able to ask 5 questions throughout the whole interview, and you must ask them one at a time. Tailor your questions to the expert's background so that you elicit as much interesting information from them as possible, and use interesting techniques to understand their current pain points. Never ask them questions about industries or experiences that seem like they are out of the expert's area of familiarity.

    Here is the goal of this design sprint, delimited below by triple backticks. You want to get information from th expert regarding this topic:

    ```
    {design_sprint_goal}
    ```

    And here is some background on the person you're talking to. Make sure you ask them interesting questions based upon their background and the information you learn from their responses:

    ```
    {expert}
    ```

    Now go ahead and begin the interview process. Remember, you are holding a dialogue with a real person. That means you should only ask one question at a time and wait for them to answer before you follow up with another question. This means each of your responses should be a SINGLE question, not a chain of questions. DO NOT ANSWER THE HUMAN RESPONSES ON YOUR OWN. Just ask a question. The format of your responses should ONLY contain a question, no other dialogue. If you do not follow these rules, you will be disqualified from the study and you will severely disappoint the researchers.

    Additionally, NEVER prepend your answer with "AI: "
    """

    INTERVIEW_HUMAN_TEMPLATE = "{question}"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(INTERVIEW_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(INTERVIEW_HUMAN_TEMPLATE),
        ]
    )

    model = Ollama(model="mistral")

    chain = prompt | model

    history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL, ttl=600)

    chain_a_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_a_with_history


# Function that invokes chain_a_with_history
def invoke_chain_a_with_history(
    chain: RunnableWithMessageHistory,
    session_id: str,
    design_sprint_goal: str = "Default Goal",
    expert: str = "Journalist highly versed in this topic",
    question: str = "What would you like to ask me?",
):
    response = chain.invoke(
        {
            "design_sprint_goal": design_sprint_goal,
            "expert": expert,
            "question": question,
        },
        config={"configurable": {"session_id": session_id}},
    )

    return response


# Function the operates the conversation chain
def operate_conversation_chain(
    design_sprint_goal: str = "Default Goal",
    expert: str = "Journalist highly versed in this topic",
):
    SESSION_A = "abc123"
    SESSION_B = "def456"

    CHAIN_A = initialize_conversation_chain_a(session_id=SESSION_A)

    STARTER_QUESTION = invoke_chain_a_with_history(
        chain=CHAIN_A,
        session_id=SESSION_A,
        design_sprint_goal=design_sprint_goal,
        expert=expert,
        question="What would you like to ask me?",
    )

    return STARTER_QUESTION
