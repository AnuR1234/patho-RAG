# Embeddings/compare_embeddings_pdf_files.py
from load_embedding_models import initialize_embedding_models, load_json_data
from pinecone_client import initialize_pinecone
import torch

def upsert_records_to_pinecone(index, data, dense_model, sparse_model, tokenizer, device):
    records = []  # This will store the records ready to be upserted into Pinecone
    for item in data:
        dense_vec = dense_model.encode(item['article-text']).tolist()
        input_ids = tokenizer(item['article-text'], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            sparse_vec = sparse_model(d_kwargs=input_ids.to(device))['d_rep'].squeeze()
        indices = sparse_vec.nonzero().squeeze().cpu().tolist()
        values = sparse_vec[indices].cpu().tolist()
        record = {
            'id': item['id'],
            'values': dense_vec,
            'metadata': item['metadata'],
            'sparse_values': {'indices': indices, 'values': values}
        }
        records.append(record)
    # Batch upsert to Pinecone
    index.upsert(vectors=records)

# def main():
#     json_file_path = '/content/drive/MyDrive/Pratikum/data/simplemed-v2_with_questions-p1.json'
#     data = load_json_data(json_file_path)
    
#     dense_model, sparse_model, tokenizer, device = initialize_embedding_models()
    
#     api_key = "your_pinecone_api_key"  # Replace with your actual Pinecone API key
#     index_name = "patho-splade"
#     index = initialize_pinecone(api_key=api_key, index_name=index_name, dimension=768, metric='cosine')
    
#     upsert_records_to_pinecone(index, data, dense_model, sparse_model, tokenizer, device)

# if __name__ == "__main__":
#     main()
