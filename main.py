import numpy as np
import faiss
import PyPDF2
from transformers import BertTokenizer, BertModel
import torch
import streamlit as st

def text_to_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(1).detach().numpy().flatten()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

st.title("PDF Content Search with Q&A Functionality")
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')

if uploaded_files:
    all_text = []
    for uploaded_file in uploaded_files:
        try:
            # Read PDF file from in-memory data
            reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(reader.pages)
            file_text = []
            for page in range(num_pages):
                file_text.append(reader.pages[page].extract_text())
            all_text.extend(file_text)
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {str(e)}")

    # Combine and index all text
    full_text = '\n'.join(all_text)
    segments = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
    dimension = 768
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array([text_to_vector(segment, tokenizer, model) for segment in segments])
    index.add(vectors)
    
    question = st.text_input("Enter your question:", "")
    if st.button("View Answer"):
        query_vector = text_to_vector(question, tokenizer, model)
        D, I = index.search(np.array([query_vector]), 1)
        if I.size > 0:
            result_index = I.flatten()[0]
            answer = segments[result_index]
            st.write("### Answer")
            st.write(answer)
        else:
            st.error("No relevant documents found.")
else:
    st.write("Please upload at least one PDF file.")


# import os
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from transformers import BertTokenizer, BertModel
# import torch
# import streamlit as st

# # Set environment variable to handle OpenMP error if not already set in the environment
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# load_dotenv()
# # You can list multiple PDF files here
# pdf_files = [
#     os.getenv("FILE_PATH_1", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf"),
#     os.getenv("FILE_PATH_2", "C:\\Users\\tmax\\OneDrive\\Desktop\\2.pdf")
# ]

# all_text = []

# for file_path in pdf_files:
#     if not os.path.isfile(file_path):
#         st.error(f"File not found: {file_path}")
#         continue
    
#     loader = PyPDFLoader(file_path=file_path)
#     document = loader.load()
    
#     if isinstance(document, list):
#         for page in document:
#             page_text = page.page_content.strip()  # Trim whitespace to clean up text
#             if page_text:  # Only add non-empty text
#                 all_text.append(page_text)
#     else:
#         st.error(f"Document format not supported or is empty for file: {file_path}")

# # Join all text from all PDFs
# full_text = '\n'.join(all_text)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def text_to_vector(text):
#     inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(1).detach().numpy().flatten()

# segments = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
# dimension = 768
# index = faiss.IndexFlatL2(dimension)
# vectors = np.array([text_to_vector(segment) for segment in segments])

# n_clusters = min(len(vectors), 10)
# if n_clusters < 10:
#     st.write(f"Reducing the number of clusters to {n_clusters} due to insufficient data points.")

# index.add(vectors)
# if len(vectors) >= 390:
#     kmeans = faiss.Kmeans(dimension, n_clusters, niter=20)
#     kmeans.train(vectors)
# else:
#     st.error("Insufficient data points for effective clustering.")

# st.title("PDF Content Search")
# query = st.text_input("Enter your search query:", "은행여신거래기본약관에서 <제1조>가 뭐야?")

# if st.button("Search"):
#     query_vector = text_to_vector(query)
#     D, I = index.search(np.array([query_vector]), 1)  # Search for the nearest neighbor
#     if I.size > 0:
#         result_index = I.flatten()[0]
#         search_result = segments[result_index]
#         st.write("### Search Result")
#         st.write(search_result)
#     else:
#         st.error("No relevant documents found.")
