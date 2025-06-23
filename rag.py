from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# below 6 lines are used for access self certified model.
import os
os.environ["CURL_CA_BUNDLE"] = ""
import urllib3
urllib3.disable_warnings()
import openai
openai.verify_ssl_certs = False

# === Load Embedding Model ===
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Load FAISS index ===
#faiss_index = FAISS.load_local("/home/sharmistha-choudhury/faiss_index_store", embedding, embedding, allow_dangerous_deserialization=True)
faiss_index = FAISS.load_local(
    folder_path="faiss_index_store",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)


# === Initialize LLM with custom endpoint ===
# here we can not used https.
llm = ChatOpenAI(
    temperature=0,
    openai_api_base="http://a7d7dc9e697d843f780df5512f98a1df-adbef44e0ec59790.elb.us-east-2.amazonaws.com/deployment/sharmistha-choudhury/mistral/v1/",  # Replace with your URL
    openai_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoic2hhcm1pc3RoYS1jaG91ZGh1cnkiLCJ0eXBlIjoiYXBpIiwiaWQiOiIvZGVwbG95bWVudC9taXN0cmFsLyJ9.ePzGazzctsADsZheMMn5vExDdI8D6AvgnhrR7_VOFC8",                 # Replace with your token
    model_name="mistralai/Mistral-7B-Instruct-v0.2"                         # Or any supported model
)

# === Setup RAG chain ===
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_index.as_retriever(),
    return_source_documents=True
)

# === Run a Query ===
query = "What sharmistha said first?"
result = rag_chain({"query": query})

# === Output Answer and Sources ===
print("Answer:")
print(result["result"])

print("\nSource Documents:")
for doc in result["source_documents"]:
    print(f"---\n{doc.page_content}")


