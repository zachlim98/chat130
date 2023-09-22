# Import necessary libraries
import streamlit as st
from streamlit_chat import message
import keys

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

# read in secret keys
gpt_key = keys.gpt_key
gpt_endpoint = "https://raid-ses-openai.openai.azure.com/"
embed_key = keys.embed_key
embed_endpoint = "https://raid-openai-e27bcf212.openai.azure.com/"

# Define the path for database load
DB_FAISS_PATH = 'dbstore'

# Load the chat model
llm = AzureChatOpenAI(openai_api_type="azure", 
                      openai_api_version="2023-05-15", 
                      openai_api_base=gpt_endpoint, 
                      openai_api_key=gpt_key, 
                      deployment_name="raidGPT", 
                      temperature=0.0)

# Define embedding model for db load
embedding_model = OpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=embed_key, 
    openai_api_base=embed_endpoint,
    openai_api_version="2023-05-15",
    deployment="text-embedding-ada-002"
    )

# Set the title for the Streamlit app
st.title("Chat with Pubs")

# load the database
db = FAISS.load_local(DB_FAISS_PATH, embedding_model)

# with st.form('myform', clear_on_submit=True):
#     page_num = st.number_input('Enter max number of pages to retrieve:', placeholder = '1 - 15', min_value=0, max_value=15, step=1)
#     submitted = st.form_submit_button('Submit')

# Create a conversational chain
retriever = db.as_retriever(search_kwargs = {"k": 10})

memory = ConversationSummaryMemory(llm=llm,
                                memory_key="chat_history", 
                                input_key="question", 
                                output_key="answer", 
                                return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                        retriever=retriever, 
                                        return_source_documents=True, 
                                        memory = memory)

# Function for conversational chat
def conversational_chat(query):
    result = qa({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! " + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Ask me anything about BWC!", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")