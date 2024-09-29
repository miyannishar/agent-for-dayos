# ------------------------------------------------------
# Streamlit
# Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗
# ------------------------------------------------------

import boto3
import uuid
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock
import streamlit as st
from langchain.prompts import PromptTemplate
import traceback

st.title("Dayos AI Assistant")

# Initialize the Bedrock runtime client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=st.secrets["aws_credentials"]["AWS_REGION"],
    aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
)

# Initialize the Knowledge Base Retriever
try:
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=st.secrets["bedrock_agent"]["BEDROCK_AGENT_ID"],
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )
except Exception as e:
    st.error(f"Error initializing retriever: {str(e)}")
    st.error(traceback.format_exc())

# Initialize the LLM with system message
system_message = """You are Dayos Agent. Dayos is a new company. You are informed about Dayos and the documentation of Oracle. Speak as a knowledgeable peer, using a straightforward and slightly irreverent tone that challenges the status quo. Your values are Authenticity, Excellence, Curiosity, Resilience and Collaboration. Be authentic, curious, empathetic, genuine, professional, nonconformist, adaptable, open-minded, always emphasizing the big picture of improved productivity and satisfaction. Use clear language, avoid jargon. Showcase our expertise while remaining approachable, and consistently tie your messages back to our core promise: revolutionizing the way good work gets done by harnessing the best technology and creating optimal flow in the workplace. Don't say that I am an AI agent. Say I am Dayos Agent. If something technical is asked and you know the answer, answer it straight. If you don't know the answer, say you don't know. Don't make up an answer. Don't say I would respond like this and all."""

model_kwargs = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 3000,
}
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime,
    model_kwargs=model_kwargs,
    streaming=False
)

# Create a custom prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{system_message}\n\nContext: {{context}}\n\nHuman: {{question}}\n\nAssistant:"
)

# Create the RetrievalQA chain with the custom prompt
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about Dayos?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Use the RetrievalQA chain to get the response
    response = qa(prompt)

    assistant_response = response['result']

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
