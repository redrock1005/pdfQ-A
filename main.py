import os
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from transformers import BertTokenizer, BertModel
import torch
import streamlit as st

# Set environment variable to handle OpenMP error if not already set in the environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv()
# You can list multiple PDF files here
pdf_files = [
    os.getenv("FILE_PATH_1", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf"),
    os.getenv("FILE_PATH_2", "C:\\Users\\tmax\\OneDrive\\Desktop\\2.pdf")
]

all_text = []

for file_path in pdf_files:
    if not os.path.isfile(file_path):
        st.error(f"File not found: {file_path}")
        continue
    
    loader = PyPDFLoader(file_path=file_path)
    document = loader.load()
    
    if isinstance(document, list):
        for page in document:
            page_text = page.page_content.strip()  # Trim whitespace to clean up text
            if page_text:  # Only add non-empty text
                all_text.append(page_text)
    else:
        st.error(f"Document format not supported or is empty for file: {file_path}")

# Join all text from all PDFs
full_text = '\n'.join(all_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(1).detach().numpy().flatten()

segments = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
dimension = 768
index = faiss.IndexFlatL2(dimension)
vectors = np.array([text_to_vector(segment) for segment in segments])

n_clusters = min(len(vectors), 10)
if n_clusters < 10:
    st.write(f"Reducing the number of clusters to {n_clusters} due to insufficient data points.")

index.add(vectors)
if len(vectors) >= 390:
    kmeans = faiss.Kmeans(dimension, n_clusters, niter=20)
    kmeans.train(vectors)
else:
    st.error("Insufficient data points for effective clustering.")

st.title("PDF Content Search")
query = st.text_input("Enter your search query:", "은행여신거래기본약관에서 <제1조>가 뭐야?")

if st.button("Search"):
    query_vector = text_to_vector(query)
    D, I = index.search(np.array([query_vector]), 1)  # Search for the nearest neighbor
    if I.size > 0:
        result_index = I.flatten()[0]
        search_result = segments[result_index]
        st.write("### Search Result")
        st.write(search_result)
    else:
        st.error("No relevant documents found.")
