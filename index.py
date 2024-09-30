# ------------------------------------------------------
# Streamlit
# Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗
# ------------------------------------------------------

import boto3
import uuid
from langchain_community.retrievers.bedrock import AmazonKnowledgeBasesRetriever
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

# Initialize the Bedrock agent runtime client
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=st.secrets["aws_credentials"]["AWS_REGION"],
    aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
)

def getAnswers(question):
    try:
        knowledgeBaseResponse = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': question},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': st.secrets["bedrock_agent"]["KNOWLEDGE_BASE_ID"],
                    'modelArn': 'arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-1-405b-instruct-v1:0'
                },
                'type': 'KNOWLEDGE_BASE'
            })
        return knowledgeBaseResponse['output']['text']
    except Exception as e:
        st.error(f"Error in getAnswers: {str(e)}")
        return None

# Initialize the LLM with system message
system_message = """You are Dayos Agent. Dayos is a new company. You are informed about Dayos and the documentation of Oracle. Speak as a knowledgeable peer, using a straightforward and slightly irreverent tone that challenges the status quo. Your values are Authenticity, Excellence, Curiosity, Resilience and Collaboration. Be authentic, curious, empathetic, genuine, professional, nonconformist, adaptable, open-minded, always emphasizing the big picture of improved productivity and satisfaction. Use clear language, avoid jargon. Showcase our expertise while remaining approachable, and consistently tie your messages back to our core promise: revolutionizing the way good work gets done by harnessing the best technology and creating optimal flow in the workplace. Don't say that I am an AI agent. Say I am Dayos Agent. If something technical is asked and you know the answer, answer it straight. If you don't know the answer, say you don't know. Don't make up an answer. Don't say I would respond like this and all. Do not mention actions like clears throat or any other actions. Just answer the question."""

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

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about Dayos?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Use the getAnswers function to get the response
    kb_response = getAnswers(prompt)
    
    if kb_response:
        # Use the LLM to generate a final response based on the KB response and system message
        final_prompt = f"{system_message}\n\nContext: {kb_response}\n\nHuman: {prompt}\n\nAssistant:"
        assistant_response = llm.predict(final_prompt)
    else:
        assistant_response = "I'm sorry, I couldn't retrieve the information at the moment. Please try again later."

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
