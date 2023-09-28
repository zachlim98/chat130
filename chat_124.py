# Import necessary libraries
import streamlit as st
# import keys

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate

from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# read in secret keys if local file 
# gpt_key = keys.gpt_key

# read in secret keys for deployment
gpt_key = st.secrets["gpt_key"]

# define the endpoints 
gpt_endpoint = "https://raid-ses-openai.openai.azure.com/"

# Define the path for database load
DB_FAISS_PATH = 'vecstore_124'

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
    openai_api_key=gpt_key, 
    openai_api_base=gpt_endpoint,
    openai_api_version="2023-05-15",
    deployment="swiftfaq-ada002"
    )

# Set the title for the Streamlit app
st.title("Chat with Pubs")

# load the database
db = FAISS.load_local(DB_FAISS_PATH, embedding_model)

# Create a conversational chain
retriever = db.as_retriever(search_kwargs = {"k": 10})

custom_template = """
You are a bot designed to answer military helicopter pilot trainees' questions from various flying handbooks and rulebooks. Use the context provided below as well as your knowledge of the AP3456 document to answer their questions. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Additionally, this was the chat history of your conversation with the user.
{chat_history}

Question: {question}

"""

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Ask me anything from the EEM and AP3456 - please give me some time, I am running on a free server")

PROMPT = PromptTemplate.from_template(template=custom_template)

memory = ConversationSummaryBufferMemory(llm=llm,
                                memory_key="chat_history", 
                                chat_memory=msgs,
                                input_key="question",
                                output_key="answer", 
                                return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                           retriever=retriever, 
                                           return_source_documents=True, 
                                           memory = memory,
                                           combine_docs_chain_kwargs={"prompt" : PROMPT})

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    with st.spinner("🤖 Bot is thinking..."):
        response = qa(prompt)["answer"]
    st.chat_message("ai").write(response)