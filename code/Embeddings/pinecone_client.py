import os
from pinecone import Pinecone, PodSpec

# Fetch the Pinecone API key from environment variable or directly assign it here
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'ff083132-7f5a-48fe-8fbc-2f8be70c5ca3')
index_name = 'patho-index'

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment="gcp-starter")

def create_index(index_name, dimension=768, metric="dotproduct"):
    """
    Creates a Pinecone index if it does not already exist.
    
    :param index_name: Name of the Pinecone index to create.
    :param dimension: Dimension of vectors to be indexed.
    :param metric: Metric used for indexing and querying vectors.
    """
    if index_name not in pinecone_client.list_indexes():
        # Create the index with specified dimension and metric
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
        )
        print(f"Created Pinecone index '{index_name}'.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")

def delete_index(index_name):
    """
    Deletes a specified Pinecone index.
    
    :param index_name: Name of the Pinecone index to delete.
    """
    if index_name in pinecone_client.list_indexes():
        pinecone_client.delete_index(index_name)
        print(f"Deleted Pinecone index '{index_name}'.")
    else:
        print(f"Pinecone index '{index_name}' does not exist.")

# Example usage
if __name__ == "__main__":
    # Optionally, create a new index
    create_index(index_name, dimension=768, metric="dotproduct")
    
    # Describe the index
    index_info = pinecone_client.describe_index(index_name)
    print(index_info)
    
    # Optionally, delete an index
    # delete_index(index_name)
