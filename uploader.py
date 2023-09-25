import streamlit as st
import os
import shutil
import datetime
# import keys

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# read in secret keys if local file 
# gpt_key = keys.gpt_key
# embed_key = keys.embed_key

# read in secret keys for deployment
gpt_key = st.secrets["gpt_key"]
embed_key = st.secrets["embed_key"]

# define the endpoints 
gpt_endpoint = "https://raid-ses-openai.openai.azure.com/"
embed_endpoint = "https://raid-openai-e27bcf212.openai.azure.com/"

UPLOAD_DIRECTORY = "uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def save_uploaded_file(uploaded_file):
    base_name = get_base_filename(uploaded_file.name)
    
    existing_file_to_delete = None
    for existing_file in os.listdir(UPLOAD_DIRECTORY):
        if existing_file.startswith(base_name):
            existing_file_to_delete = existing_file
            break

    # If an existing file with the same base name is found, delete it
    if existing_file_to_delete:
        os.remove(os.path.join(UPLOAD_DIRECTORY, existing_file_to_delete))

    # Now save the new file with its original name
    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def list_files(directory):
    files = os.listdir(directory)
    if len(files) == 0:
        return []
    return sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

def get_upload_time(file_path):
    timestamp = os.path.getmtime(file_path)
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

def extract_version(filename):
    base_name = os.path.splitext(filename)[0]
    version_str = base_name.split("AL")[-1] if "AL" in base_name else None
    return version_str if version_str else "Unknown"

def get_base_filename(filename):
    return filename.split("AL")[0] if "AL" in filename else filename

def load_docs(upload_dir):
    loader = DirectoryLoader(upload_dir, glob='**/*.pdf', loader_cls=PyPDFLoader)

    docs = loader.load()
    
    return docs

def vector_load(docs, key, endpoint):
    
    embedding_model = OpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=key, 
    openai_api_base=endpoint,
    openai_api_version="2023-05-15",
    deployment="text-embedding-ada-002"
    )
    
    # need to build some logic here for checking the database - if exists then just add if not, create

    db = FAISS.from_documents(docs, embedding_model)

    return db

# function to combine both
def create_embed(upload_dir):
    docs = load_docs(upload_dir)
    db = vector_load(docs, embed_key, embed_endpoint)
    
    db.save_local("vecstore")
    
    return db

st.title("Pubs Upload Version Tracker ✈️")

uploaded_file = st.file_uploader("Upload a file 💻", accept_multiple_files=True)
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.write(f"Saved {file_path}")

st.write("Uploaded Files:")

files = list_files(UPLOAD_DIRECTORY)

# Group files by their base name and keep only the latest version for each base name
latest_files = {}
for file in files:
    base_name = get_base_filename(file)
    if base_name not in latest_files:
        latest_files[base_name] = file

if latest_files:
    file_data = [{"Name": get_base_filename(f), "Version": extract_version(f), "Upload Date": get_upload_time(os.path.join(UPLOAD_DIRECTORY, f))} for f in latest_files.values()]
    st.table(file_data)
else:
    st.write("No files uploaded yet.")

# If you want to allow deletion (optional)
if st.button("Clear all files"):
    shutil.rmtree(UPLOAD_DIRECTORY)
    os.makedirs(UPLOAD_DIRECTORY)

st.write("Once all the files you want to upload are done, embed them into the database")

if st.button("Embed files"):

# Show progress bar while embedding
    progress_bar = st.progress(0)
    st.write("Embedding the file...")
    
    # Embed the file (replace this with actual embedding process and update progress as needed)
    create_embed(UPLOAD_DIRECTORY)
    progress_bar.progress(100) # Assuming the embedding process is complete
    
    st.write(f"Saved and embedded all files from {UPLOAD_DIRECTORY}")