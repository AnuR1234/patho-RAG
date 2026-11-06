# Evaluation/patho_rag_test_data.py
import json
from pinecone import Pinecone, PodSpec
from Embeddings.load_embedding_models import initialize_embedding_models, load_json_data
from Embeddings.pinecone_client import index_name,pinecone_client
import torch
from tqdm.auto import tqdm

json_file_path = '/path/to/simplemed-v2_with_questions-p1.json'

def builder(records, dense_model, sparse_model, tokenizer, device):
    upserts = []
    for record in records:
        article_id = str(record['id'])
        for chunk_id, chunk_text in enumerate(record['article-chunks']):
            unique_id = f"{article_id}#chunk:{chunk_id}"

            # Embed the single chunk text
            dense_vec = dense_model.encode(chunk_text, convert_to_tensor=True).cpu().tolist()

            # Tokenize and embed the single chunk text using sparse_model
            input_ids = tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                sparse_vec = sparse_model(d_kwargs=input_ids.to(device))['d_rep'].squeeze()

            # Process sparse vector
            indices = sparse_vec.nonzero().squeeze().cpu().tolist() if sparse_vec.dim() > 0 else []
            values = sparse_vec[indices].cpu().tolist() if indices else []
            sparse_values = {"indices": indices, "values": values}

            # Prepare metadata
            metadata = {
                'source': record['metadata']['source'],
                'article_title': record['metadata']['article-title'],
                'questions': record['questions'],
                'correct_answers': record['correct-answers']
            }

            # Prepare upsert object for Pinecone
            upsert = {
                'id': unique_id,
                'values': dense_vec,
                'sparse_values': sparse_values,
                'metadata': metadata
            }
            upserts.append(upsert)
    return upserts

def main():
    # Load test data
    data = load_json_data(json_file_path)

    # Initialize models and Pinecone client
    dense_model, sparse_model, tokenizer, device = initialize_embedding_models()
    index = pinecone_client.Index(index_name)

    # Upsert records into Pinecone in batches
    batch_size = 32
    for i in tqdm(range(0, len(data), batch_size)):
        end_i = min(i + batch_size, len(data))
        batch = data[i:end_i]
        upserts = builder(batch, dense_model, sparse_model, tokenizer, device)
        index.upsert(vectors=upserts)
        print(f"Upserted batch {i//batch_size + 1}/{len(data)//batch_size + 1}")

if __name__ == "__main__":
    main()
