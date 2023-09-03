import chainlit as cl
import openai 
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Setting the openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")


@cl.on_message
async def main(message : str):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role":"assistant", "content":"you are a helpful assistant"},
                    {"role":"user", "content":message}
                    ],
        temperature = 0.3,
    )
    await cl.Message(content = response['choices'][0]['message']['content']).send()
