import streamlit as st
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
import toml

#streamlit run app.py


# Load the configuration file


# cluster connection
client = QdrantClient(
    url = st.Secrets["url"] ,
    port= st.Secrets["port"],
    api_key = st.Secrets["api_key"]
)

vectors_config = models.VectorParams(
    size = 1536,
    distance = models.Distance.COSINE
)

OPENAI_API_KEY= st.Secrets["openai_key"]

embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # passs the key in here 

vector_store = Qdrant(
    client=client, 
    collection_name="MyCollection", 
    embeddings=embeddings,
)

# the chunking is a one time thing and only needs to be done when training
# data = pd.read_csv('cleaned_movie_data.csv')
# data = data.dropna()
# data = data[:1000]
# data = data.reset_index(drop = True)

# chunks = []
# for i in range(0, len(data)):
#     chunk = ',' .join(f"{col}: {data.iloc[i][col]}" for col in data.columns)
#     chunks.append(chunk)

# vector_store.add_texts(chunks)

retriever = vector_store.as_retriever()

template = """You are a movie expert and using the {context} answer the user question {question}"""
prompt = PromptTemplate(template= template, input_variables=["question", "context"])

model = ChatOpenAI(model="gpt-3.5-turbo", api_key = OPENAI_API_KEY)
ouput_parser = StrOutputParser()

retriever_2 = vector_store.as_retriever()

rag_chain2 = (
    {
        "context": retriever_2,
        "question": RunnablePassthrough()
     }
     |prompt
     |model
     |StrOutputParser()
)

st.title("Movie Expert")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = rag_chain2.invoke(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# response = f"Echo: {prompt}"
# if prompt:
    