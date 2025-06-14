import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexai-client-api.json"

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

# Load embedding model
embedding = VertexAIEmbeddings(model_name="text-embedding-004")

pdf_folder = "papers/"
db_path = "papers_db"

# Load all PDF 
all_documents = []
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'): 
        print(f"Processing: {filename}")
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata['title'] = filename.replace(".pdf", "")
        all_documents.extend(docs)

print(f"Total documents loaded: {len(all_documents)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 150)
chunks = text_splitter.split_documents(all_documents)

# store embeddings in Chroma
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=db_path
)

print("Vector Database Created.")