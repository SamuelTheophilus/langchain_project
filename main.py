import os
import chainlit as cl
import openai 
from dotenv import load_dotenv
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader , TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load Environment Variables
load_dotenv()
# Setting the openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
embeddings = OpenAIEmbeddings()

welcome = """Hi, this is my PDF QA demo.
1. Please upload a file.
2. Ask Questions about the file.
"""

def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain": Loader = TextLoader
    if file.type == "application/pdf": Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False) as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents= documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

    return docs

def get_docsearch(file: AskFileResponse):
    docs = process_file(file)
    cl.user_session.set("docs", docs)
    return Chroma.from_documents(docs, embeddings)


@cl.on_chat_start
async def start():
    await cl.Message(content= "You can now chat with your pdf").send()
    files = None 
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome,
            accept=["text/plain", "application/pdf"],
            max_size_mb= 110,
            timeout= 10000
        ).send()
    
    file = files[0]
    msg = cl.Message(content= f"Processing {file.name}")
    await msg.send()


    docsearch = await cl.make_async(get_docsearch)(file)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature= 0.1, streaming=True),
        chain_type="stuff",
        retriever = docsearch.as_retriever(max_tokens_limit = 4097)
    )


    msg.content = f"{file.name} now processed, you can ask your questions"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message : str):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer= True , answer_prefix_tokens= ["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks =[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []


    # Retrieve the documents from the user session.
    documents = cl.user_session.get("docs")
    metadata = [doc.metadata for doc in documents]
    all_sources = [m["source"] for m in metadata]

    if sources:
        found_sources = []

        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue

            text = documents[index].page_content
            found_sources.append(source_name)


            # Create text element referenced in the message
            source_elements.append(cl.Text(content = text, name= source_name))

        answer += f"\nSources: {', '.join(found_sources)}" if found_sources else f"\nNo sources found"
        
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else: 
        await cl.Message(content= answer, elements= source_elements).send()

