def load_docs(filepath):
    loader = DirectoryLoader(filepath, glob='**/*.pdf', loader_cls=PyPDFLoader)

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

# Create a file uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload File", type="pdf", accept_multiple_files= True)

# Handle file upload
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data using CSVLoader
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store and save embeddings
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)