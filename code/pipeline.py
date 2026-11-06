import sys
import os
import json

# Assuming the 'data_prep' directory and 'pdf_details.json' are located relative to this script's location
# Adjust these paths as necessary based on your actual directory structure and where you place 'pdf_details.json'
data_prep_path = os.path.join(os.path.dirname(__file__), 'data', 'Data_prep')
pdf_details_path = os.path.join(os.path.dirname(__file__), 'pdf_details.json')
embeddings_path = os.path.join(os.path.dirname(__file__), 'Embeddings')
# Add the 'data_prep' directory to the system path to import the modules
sys.path.append(data_prep_path)
sys.path.append(embeddings_path)

from data_ingestion import load_pdfs
from data_preprocessing import chunk_documents, visualize_token_counts, save_documents_to_json
# Importing embedding modules
from load_embedding_models import initialize_embedding_models, load_json_data
from pinecone_client import initialize_pinecone
from compare_embeddings_pdf_files import upsert_records_to_pinecone

def load_pdf_details(json_path):
    """
    Load PDF details from a JSON file.
    :param json_path: Path to the JSON file containing PDF details.
    :return: Dictionary with PDF paths and their start pages.
    """
    with open(json_path, 'r') as file:
        return json.load(file)

def main():
    # Load PDF details from the JSON file
    pdf_details = load_pdf_details(pdf_details_path)

    # Load and process PDFs into Document objects
    documents = load_pdfs(pdf_details)
    print(f"Loaded {len(documents)} documents from specific pages.")

    # Chunk documents
    chunked_documents = chunk_documents(documents, chunk_size=1000, overlap=200)
    print(f"Chunked into {len(chunked_documents)} documents.")

    # Visualize token counts
    visualize_token_counts(chunked_documents)

    # Save chunked documents to JSON
    output_json_path = os.path.join(os.path.dirname(__file__), 'ProcessedDocuments.json')
    save_documents_to_json(chunked_documents, output_json_path)
    print(f"Chunked documents saved to {output_json_path}")
    
    # Initialize embedding models
    dense_model, sparse_model, tokenizer, device = initialize_embedding_models()

    # Initialize Pinecone client
    api_key = "your_pinecone_api_key"  # Replace with your actual Pinecone API key
    index_name = "patho-splade"
    index = initialize_pinecone(api_key=api_key, index_name=index_name)

    # Upsert chunked documents into Pinecone index
    upsert_records_to_pinecone(index, chunked_documents, dense_model, sparse_model, tokenizer, device)

    print(f"Documents have been processed and embedded into Pinecone index: {index_name}")

if __name__ == "__main__":
    main()
