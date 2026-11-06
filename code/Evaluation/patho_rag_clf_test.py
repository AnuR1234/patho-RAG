# Evaluation/patho_rag_clf_test.py
import sys
import os

import sys
import os

# Specify the direct path to the file
file_path = '/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code/Embeddings/load_embedding_models.py'

# Check if the file exists at the specified path
if os.path.exists(file_path):
    print(f"The file exists: {file_path}")
else:
    print(f"File not found: {file_path}")
    
# Correctly identify the 'Embeddings' directory path
embeddings_directory = '/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code/Embeddings'
print("Embeddings directory:", embeddings_directory)

# Add the 'Embeddings' directory to the Python path
if embeddings_directory not in sys.path:
    sys.path.append(embeddings_directory)
from load_embedding_models import initialize_embedding_models
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from pinecone_client import index_name,pinecone_client
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class PathoRAG:
    def __init__(self, dense_model, sparse_model, tokenizer, index_name,articles_data):
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.tokenizer = tokenizer
        # Assume `index` is your Pinecone client initialized and ready to query
        self.index = pinecone_client.Index(index_name)
        self.articles_data = articles_data

    def encode(self, text):
        # Create dense embedding
        dense_vec = self.dense_model.encode(text).tolist()

        # Create sparse embedding
        input_ids = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = self.sparse_model(**input_ids)
        vec = torch.max(
            torch.log(1 + torch.relu(output.logits)) * input_ids['attention_mask'].unsqueeze(-1),
            dim=1)[0].squeeze()

        # Assuming the sparse vector 'vec' should be treated similarly to 'sparse_vec' in your original code
        # Convert to Pinecone compatible format
        indices = torch.nonzero(vec, as_tuple=False).squeeze().cpu().tolist()
        values = vec[indices].cpu().tolist() if indices else []
        sparse_dict = {"indices": indices, "values": values}

        return dense_vec, sparse_dict


    def hybrid_query(self, query_text, alpha=0.5):
        dense_vec, sparse_dict = self.encode(query_text)

        # Scale the vectors for hybrid search
        hybrid_dense = [val * alpha for val in dense_vec]
        hybrid_sparse = {"indices": sparse_dict["indices"], "values": [val * (1 - alpha) for val in sparse_dict["values"]]}

        # Execute the query
        result = self.index.query(
            vector=hybrid_dense,
            sparse_vector=hybrid_sparse,
            top_k=10,
            include_metadata=True
        )

        return result
    def predict(self, question):
        # Use hybrid_query to get results for the question
        result = self.hybrid_query(question, alpha=0.5)  # Adjust alpha as needed

        # Process results to get the most similar article chunk ID
        if result["matches"]:
            top_match_id = result["matches"][0]["id"]
            return top_match_id
        else:
            return None

    def fetch_article_context(self, match_id):
        """
        Fetches the article's chunks and title for the given match ID.
        """
        parts = match_id.split('#')
        #print(f'the parts are {parts}')
        article_id = '#'.join(parts[:2])  # Adjust this if necessary to match your data structure
        #print(f'The article id is {article_id}')

        # Iterate over the list to find the article by ID
        for article in self.articles_data:
            if article['id'] == article_id:
                #print('Article found.')
                article_title = article['metadata']['article-title']
                #print(f'the article_title is {article_title}')
                # Concatenate all chunks into a single string for context
                article_chunks = "\n".join([f"{cid}: {chunk}" for cid, chunk in article['article-chunks'].items()])
                context = f"Article Title: {article_title}\nArticle Chunks:\n{article_chunks}"
                #print(f'the context was {context}')
                return context

        return "Article context not found.", "No relevant article found."



class ClassificationTest:
    def __init__(self, patho_rag, articles_data):
        self.patho_rag = patho_rag
        self.articles_data = articles_data
        self.articles_true = []
        self.questions = []
        self.articles_predicted = []

    def run_test(self):
        for article in self.articles_data:
            for q in article["questions"]:
                self.questions.append(q)
                self.articles_true.append(article["id"])  # Assuming 'id' is the article identifier

        for question in self.questions:
            top_match_id = self.patho_rag.predict(question)
            if top_match_id:
                predicted_article_id = top_match_id.split("#chunk:")[0]
                self.articles_predicted.append(predicted_article_id)
            else:
                self.articles_predicted.append(None)  # Handle case where no prediction is made

        self.calculate_metrics()

    def calculate_metrics(self):
        encoder = LabelEncoder()
        articles_true_encode = encoder.fit_transform(self.articles_true)
        articles_predicted_encode = encoder.transform(self.articles_predicted)

        accuracy = accuracy_score(articles_true_encode, articles_predicted_encode)
        precision = precision_score(articles_true_encode, articles_predicted_encode, average='weighted')
        recall = recall_score(articles_true_encode, articles_predicted_encode, average='weighted')
        confusion_mat = confusion_matrix(articles_true_encode, articles_predicted_encode)

        print(f"Accuracy = {accuracy:.2f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Confusion Matrix:")
        print(confusion_mat)
        self.visualize_metrics()
    def visualize_metrics(self):
        encoder = LabelEncoder()
        articles_true_encode = encoder.fit_transform(self.articles_true)
        articles_predicted_encode = encoder.transform(self.articles_predicted)

        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(articles_true_encode, articles_predicted_encode, average=None)

        # Mapping class numbers back to article IDs
        article_ids = [encoder.inverse_transform([idx])[0] for idx in range(len(encoder.classes_))]

        # Plot metrics
        self.plot_metric(article_ids, precision, 'Precision')
        self.plot_metric(article_ids, recall, 'Recall')
        self.plot_metric(article_ids, f1, 'F1-Score')

        # Generate and plot confusion matrix
        confusion_mat = confusion_matrix(articles_true_encode, articles_predicted_encode)
        self.plot_confusion_matrix(confusion_mat, article_ids)

    def plot_metric(self, article_ids, values, title):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(article_ids)), values, color='skyblue')
        plt.xlabel('Article ID')
        plt.ylabel(title)
        plt.title(f'{title} for each Article')
        plt.xticks(range(len(article_ids)), article_ids, rotation=45, ha='right')
        plt.show()

    def plot_confusion_matrix(self, confusion_mat, article_ids):
        plt.figure(figsize=(12, 12))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=article_ids, yticklabels=article_ids)
        plt.ylabel('Actual Article ID')
        plt.xlabel('Predicted Article ID')
        plt.title('Confusion Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, ha='right')
        plt.tight_layout()
        plt.show()