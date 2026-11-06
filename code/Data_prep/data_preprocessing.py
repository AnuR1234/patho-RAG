# data_preprocessing.py
import json
from data_ingest import Document, process_pdf  # Adjust import path as needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import seaborn as sns
from tiktoken import tiktoken  # Make sure this import matches the library's structure

def chunk_documents(documents, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, overlap, len, ['\n\n', '\n', ' ', ''])
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_documents.append(Document(chunk, doc.metadata))
    return chunked_documents

def visualize_token_counts(documents):
    # Assuming tiktoken_len is defined or imported
    token_counts = [tiktoken_len(doc.page_content) for doc in documents]
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 6))
    sns.histplot(token_counts, kde=False, bins=50)
    plt.title("Token Counts Histogram")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.show()

def save_documents_to_json(documents, file_path):
    docs_dict = [doc.to_dict() for doc in documents]
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(docs_dict, file, ensure_ascii=False, indent=4)

def tiktoken_len(text):
    # Placeholder for tiktoken length calculation
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)
