from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load documents (can be PDFs, txt, etc.)
loader = TextLoader("absolute path of the file")
docs = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Use an embedding model (e.g., all-MiniLM)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the FAISS index
faiss_index = FAISS.from_documents(documents, embedding)

# Save index (optional)
faiss_index.save_local("faiss_index_store")
