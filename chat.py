# Import necessary libraries
import streamlit as st
from streamlit_chat import message
# import keys

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate

# read in secret keys if local file 
# gpt_key = keys.gpt_key
# embed_key = keys.embed_key

# read in secret keys for deployment
gpt_key = st.secrets["gpt_key"]
embed_key = st.secrets["embed_key"]

# define the endpoints 
gpt_endpoint = "https://raid-ses-openai.openai.azure.com/"
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

# Create a conversational chain
retriever = db.as_retriever(search_kwargs = {"k": 10})

custom_template = """
You are a bot designed to answer military pilot trainees' questions from various flying handbooks and rulebooks. Use the context provided below to answer their questions. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Additionally, this was the chat history of your conversation with the user.
{chat_history}

Question: {question}

"""

PROMPT = PromptTemplate.from_template(template=custom_template)

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
    return result["answer"], result["source_documents"][0].metadata["source"]

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
        output, source = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output + "\n\n"  + "Source: " + source)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")