import streamlit as st
import boto3
import json
import uuid

st.title("Dayos AI Assistant")

bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=st.secrets["aws_credentials"]["AWS_REGION"],
    aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
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

    response = bedrock_agent_runtime.invoke_agent(
        agentId=st.secrets["bedrock_agent"]["BEDROCK_AGENT_ID"],
        agentAliasId=st.secrets["bedrock_agent"]["BEDROCK_AGENT_ALIAS_ID"],
        sessionId=st.session_state.session_id,
        inputText=prompt
    )

    assistant_response = ""
    for event in response['completion']:
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                assistant_response += chunk['bytes'].decode('utf-8')

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})