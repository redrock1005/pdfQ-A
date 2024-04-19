import os
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from transformers import BertTokenizer, BertModel
import torch

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
        print(f"File not found: {file_path}")
        continue
    
    loader = PyPDFLoader(file_path=file_path)
    document = loader.load()
    
    if isinstance(document, list):
        for page in document:
            page_text = page.page_content.strip()  # Trim whitespace to clean up text
            if page_text:  # Only add non-empty text
                all_text.append(page_text)
    else:
        print(f"Document format not supported or is empty for file: {file_path}")

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
    print(f"Reducing the number of clusters to {n_clusters} due to insufficient data points.")

index.add(vectors)
if len(vectors) >= 390:
    kmeans = faiss.Kmeans(dimension, n_clusters, niter=20)
    kmeans.train(vectors)
    labels = kmeans.index.search(vectors, 1)[1]

    for i, label in enumerate(labels.flatten()):
        print(f"Segment {i+1}, Cluster {label}:")
        print(segments[i])
        print("==="*20)
else:
    print("Insufficient data points for effective clustering.")


# 검색 쿼리 실행
query = "은행여신거래기본약관에서 <제1조>가 뭐야?"
query_vector = text_to_vector(query)
D, I = index.search(np.array([query_vector]), 1)  # 1개의 가장 가까운 이웃 찾기

if I.size > 0:
    result_index = I.flatten()[0]
    search_result = segments[result_index]
    print(search_result)
else:
    print("No relevant documents found.")



#96. chunk 백터 저장함 > 임베딩 후에 저장해야되서 97로 원복
# import os
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter

# load_dotenv()
# file_path = os.getenv("FILE_PATH", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf")

# if not os.path.isfile(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# loader = PyPDFLoader(file_path=file_path)
# document = loader.load()

# if isinstance(document, list):
#     all_text = []
#     for page in document:
#         page_text = page.page_content
#         all_text.append(page_text)

#     full_text = '\n'.join(all_text)
    
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
#     # Create a Faiss index
#     dimension = 768  # Example dimension size for BERT-like models
#     index = faiss.IndexFlatL2(dimension)
    
#     # Example function to convert text to vector
#     def text_to_vector(text):
#         # This function should really use an NLP model to encode text
#         # Using random vector for demonstration purposes
#         return np.random.rand(dimension).astype('float32')
    
#     texts = text_splitter.split_text(full_text)

#     # Print each chunk and store to vector storage
#     for i, text_chunk in enumerate(texts):
#         print(f"Chunk {i+1}:\n{text_chunk}\n")
#         print("==="*20)  # Separator for readability between chunks
#         vector = text_to_vector(text_chunk)
#         index.add(np.array([vector]))  # Add vector to Faiss index

# else:
#     print("Document is not in list format or empty")


#97. pdf글자를 chunk로 쪼갬 (정상)
# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter

# load_dotenv()
# file_path = os.getenv("FILE_PATH", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf")  # Default file path

# if not os.path.isfile(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# loader = PyPDFLoader(file_path=file_path)
# document = loader.load()

# if isinstance(document, list):
#     all_text = []
#     for page in document:
#         page_text = page.page_content
#         all_text.append(page_text)

#     full_text = '\n'.join(all_text)
    
#     # Initialize CharacterTextSplitter with specified chunk size and overlap
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
#     # Print available methods and attributes
#     print(dir(text_splitter))
    
#     # Assuming the correct method is available, check the method list printed and use it here:
#     try:
#         texts = text_splitter.split_text(full_text)  # Update 'split' to the correct method if needed
        
#         # Print each chunk
#         for i, text_chunk in enumerate(texts):
#             print(f"Chunk {i+1}:\n{text_chunk}\n")
#             print("==="*20)  # Separator for readability between chunks
#     except AttributeError as e:
#         print(f"Error: {e}")
#         print("Please check the printed method list and use the correct method for splitting text.")

# else:
#     print("Document is not in list format or empty")



#98.pdf에 글자 출력하는 코드 (정상)
# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader

# load_dotenv()
# file_path = os.getenv("FILE_PATH", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf")  # Default file path

# if not os.path.isfile(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# loader = PyPDFLoader(file_path=file_path)
# document = loader.load()

# if isinstance(document, list):
#     # Initialize an empty list to collect all page contents
#     all_text = []
#     for page in document:
#         # Accessing page_content attribute to get text of each page
#         page_text = page.page_content
#         all_text.append(page_text)
    
#     # Join all texts into a single string
#     full_text = '\n'.join(all_text)
#     print(full_text)
# else:
#     print("Document is not in list format or empty")


#99. pdf에 포함된 객체 구조 확인하는 코드
#  import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader

# load_dotenv()
# file_path = os.getenv("FILE_PATH", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf")  # Default file path

# if not os.path.isfile(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# loader = PyPDFLoader(file_path=file_path)
# document = loader.load()

# if isinstance(document, list):
#     for page in document:
#         print(f"Available attributes for the page object: {dir(page)}")  # 출력을 통해 속성 확인
# else:
#     print("Document is not in list format or empty")


# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter

# load_dotenv()
# file_path = os.getenv("FILE_PATH", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf")  # Default file path

# if not os.path.isfile(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# loader = PyPDFLoader(file_path=file_path)
# document = loader.load()

# # Initialize a variable to collect all text
# full_text = ""

# # Check the structure of the loaded document
# if isinstance(document, list):
#     # If the document is a list, assume each item is a page object
#     for page in document:
#         # Check if page has a text attribute or method
#         if hasattr(page, 'text'):
#             full_text += page.text
#         else:
#             print("Page does not have a 'text' attribute. Check the page object structure.")
# else:
#     print("Loaded document does not follow expected list structure.")
#     exit()

# # Split the full text into manageable chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# texts = text_splitter.split_documents(full_text)

# # Print each chunk
# for chunk_index, chunk in enumerate(texts, start=1):
#     cleaned_chunk = chunk.replace('\n', ' ')  # Removing newlines for better readability
#     print(f"Chunk {chunk_index}: {cleaned_chunk[:200]}...")  # Print the first 200 characters of each chunk
#     print()  # Adds an empty line between chunk outputs for better readability









# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain import hub
# from langchain_community.chat_models import ChatOpenAI
# from langchain.schema.runnable import RunnablePassthrough



# load_dotenv()
# file_path = os.getenv("FILE_PATH", "C:\\Users\\tmax\\OneDrive\\Desktop\\1.pdf")  # 기본 경로 설정
# model_name = os.getenv("MODEL_NAME", "gpt-4-0613")

# if not os.path.isfile(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# loader = PyPDFLoader(file_path=file_path)
# document = loader.load()
# print(document[].page_content[:200])  # 내용 일부 출력

# # 텍스트 분할
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# texts = text_splitter.split_documents(document)

# # 임베딩 설정
# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(texts, embeddings)
# retriever = docsearch.as_retriever()

# # langchain hub에서 Prompt 다운로드
# rag_prompt = hub.pull("rlm/rag-prompt")

# # ChatGPT 모델 초기화
# llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)

# # RAG 체인 설정
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()} 
#     | rag_prompt 
#     | llm 
# )

# # 예제 질문 사용
# response = rag_chain.invoke("은행 여신 거래 기본 약관에는 어떤 내용이 포함되어 있나요?")
# print(response)
