import torch
from tqdm.auto import tqdm
from Evaluation.patho_rag_test_data import builder, load_json_data
from Evaluation.patho_rag_clf_test import PathoRAG, ClassificationTest
from Embeddings.load_embedding_models import initialize_embedding_models
from Embeddings.pinecone_client import index_name, pinecone_client

# Define the path to your test data JSON
json_file_path = '/Users/anureddy/Desktop/SEM03/Pratikums/pathlogy_pratikum/data/simplemed-v2_with_questions-p1.json'

def main():
    # Load test data
    data = load_json_data(json_file_path)

    # Initialize models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dense_model, sparse_model, tokenizer, _ = initialize_embedding_models()

    # Initialize Pinecone client and describe the index (optional step for verification)
    index = pinecone_client.Index(index_name)
    print(index.describe_index_stats())

    # Upsert test data into Pinecone in batches
    # upserts = builder(data, dense_model, sparse_model, tokenizer, device)
    # batch_size = 32
    # for i in tqdm(range(0, len(upserts), batch_size), desc='Upserting batches'):
    #     batch = upserts[i:i + batch_size]
    #     index.upsert(vectors=batch)
    # print("Completed upserting data into Pinecone.")

    # Initialize PathoRAG for queries and run classification tests
    patho_rag = PathoRAG(dense_model, sparse_model, tokenizer, index_name, data)

    # Example of querying with a question
    question = "Which type of cancer is associated with a mutation in the HER-2 gene?"
    result = patho_rag.hybrid_query(question, alpha=0.5)
    top_match_id = result["matches"][0]["id"]
    metadata = result["matches"][0]["metadata"]["article_title"]
    print(f"The top matching article chunk ID for the query is: {top_match_id}")
    print(f"The title of the top matching article is: {metadata}")

    # Run classification tests and visualize metrics
    classification_test = ClassificationTest(patho_rag, data)
    classification_test.run_test()

if __name__ == "__main__":
    main()
