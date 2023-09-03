import os
import chainlit as cl
import openai 
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain

# Load Environment Variables
load_dotenv()

# Setting the openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")


template = """
Question: {question}
Answer: let's think step by step
"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template = template, input_variables=["question"])
    llm_chain = LLMChain(
        llm= OpenAI(temperature=0.9, streaming = True),
        prompt= prompt,
        verbose= True
    )

    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message : str):
    llm_chain = cl.user_session.get("llm_chain")
    response = await llm_chain.acall(message, callbacks = [cl.AsyncLangchainCallbackHandler()])
    await cl.Message(response["text"]).send()
