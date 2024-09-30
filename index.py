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

# Initialize the Bedrock agent runtime client
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=st.secrets["aws_credentials"]["AWS_REGION"],
    aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are Dayos Agent. Dayos is a new company. You are informed about Dayos and the documentation of Oracle. Speak as a knowledgeable peer, using a straightforward and slightly irreverent tone that challenges the status quo. You may be asked about Dayos or some technical knowledg about otacle.

Human: {question}
Assistant:"""
)

def getAnswers(question):
    try:
        # Format the question using the prompt template
        formatted_question = prompt_template.format(question=question)
        
        knowledgeBaseResponse = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': formatted_question},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': "G20GV5OFLB",
                    'modelArn': 'arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-1-405b-instruct-v1:0'
                },
                'type': 'KNOWLEDGE_BASE'
            })
        return knowledgeBaseResponse['output']['text']
    except Exception as e:
        st.error(f"Error in getAnswers: {str(e)}")
        return None

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
    assistant_response = getAnswers(prompt)

    print(assistant_response)
    
    if assistant_response:
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    else:
        st.error("I'm sorry, I couldn't retrieve the information at the moment. Please try again later.")
