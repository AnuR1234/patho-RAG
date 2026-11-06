import os
import sys 
import importlib.util

import os

print("Current Working Directory:", os.getcwd())
# Construct the path to the file
project_root = '/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
file_path = '/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code/Embeddings/pinecone_client.py'

# Load the module
spec = importlib.util.spec_from_file_location("pinecone_client", file_path)
pinecone_client = importlib.util.module_from_spec(spec)




print(index_name)
import torch
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from Embeddings.pinecone_client import index_name, pinecone_client


# Add this directory to the Python path

from Evaluation.patho_rag_clf_test import PathoRAG
from Evaluation.patho_rag_test_data import load_json_data
from Embeddings.load_embedding_models import initialize_embedding_models
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Configuration and Initialization
json_file_path  = '/Users/anureddy/Desktop/SEM03/Pratikums/pathlogy_pratikum/data/simplemed-v2_with_questions-p1.json'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the template for your LLMChain prompt
template = """
Given the article chunks below, answer the following question. If the answer can be derived from the article chunks, provide a detailed explanation using only the information provided. If the article chunks do not contain enough information to answer the question directly, clearly respond with 'I don't know.'

Article Chunks:
{context}

Question:
{question}

Answer:
"""
def main():
    # Load your data
    articles_data = load_json_data(json_file_path)

    # Initialize your models and tokenizer
    # This part is pseudo-code, adjust according to your actual initialization
    dense_model, sparse_model, tokenizer, index_name = initialize_embedding_models()

    # Initialize PathoRAG with loaded data
    patho_rag = PathoRAG(dense_model, sparse_model, tokenizer, index_name, articles_data)

    # Setup LangChain LLMChain with OpenAI
    llm = OpenAI(temperature=0.7)
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    def generate_response(question):
        # Fetch context for the question
        context = patho_rag.fetch_article_context(patho_rag.predict(question))
        if context.startswith("No relevant article found"):
            print("No relevant context found for the question.")
            return

        # Generate response using LLMChain
        response = llm_chain.run({"context": context, "question": question})
        response_text = response.choices[0].text if response.choices else "No response generated."

        # Extract title and chunks for references
        article_title, article_chunks = context.split("\nArticle Chunks:\n", 1)
        article_title = article_title.replace("Article Title:", "").strip()
        
        # Prepare and print the formatted output
        formatted_output = (
            f"Question:\n{question}\n\n"
            f"Solution:\n{response_text}\n\n"
            f"References:\nArticle Title: {article_title}\nArticle Chunks:\n{article_chunks}"
        )
        print(formatted_output)

    # Example usage
    question = "Which type of cancer is associated with a mutation in the HER-2 gene?"
    generate_response(question)

if __name__ == "__main__":
    main()