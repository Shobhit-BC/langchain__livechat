from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
import os
from dotenv import load_dotenv

load_dotenv()

# initializing duckduckgo serach tool
ddg = DuckDuckGoSearchRun()

# creating serpapi serach tool
search = SerpAPIWrapper(serpapi_api_key="6721671ae902527f1dd30908e1ded3aba9a8a79bf0864179950e99a7244b6edd")
serp=Tool(
        name = "Serp Api Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    )

tools = [serp,ddg]

turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo',openai_api_key=os.getenv('OPENAI_API_KEY')
)

# creating memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)


# create our agent
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=turbo_llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)



# to end the conversation pass an empty input
while (user_input:=input("Enter your message:")) != " ":
    response=conversational_agent(user_input)
    response=response["output"]
    print(f"Bot Answer: {response}")
