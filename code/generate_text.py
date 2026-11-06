import os
import sys
import torch
import openai 
#from openai import OpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
print(f'the exe {sys.executable}')

# Assuming the necessary imports are correctly set up
from Evaluation.patho_rag_clf_test import PathoRAG
from Evaluation.patho_rag_test_data import load_json_data
from Embeddings.load_embedding_models import initialize_embedding_models
from Embeddings.pinecone_client import index_name, pinecone_client

# Configuration and Initialization
json_file_path = '/Users/anureddy/Desktop/SEM03/Pratikums/pathlogy_pratikum/data/simplemed-v2_with_questions-p1.json'
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load your data
data = load_json_data(json_file_path)

# Initialize models and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dense_model, sparse_model, tokenizer, _ = initialize_embedding_models()

# Initialize PathoRAG for queries
patho_rag = PathoRAG(dense_model, sparse_model, tokenizer, index_name, data)

# Setup LangChain LLMChain with OpenAI
llm = OpenAI(temperature=0.7)
template = """
Given the article chunks below, answer the following question. If the answer can be derived from the article chunks, provide a detailed explanation using only the information provided. If the article chunks do not contain enough information to answer the question directly, clearly respond with 'I don't know.'

Article Chunks:
{context}

Question:
{question}

Answer:
"""
prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

def get_context_and_question(question):
    """
    Fetches article chunks and title for the given question and prepares them as context.
    """
    patho_rag = PathoRAG(dense_model, sparse_model, tokenizer, index_name, data)  # Initialize PathoRAG here
    match_id = patho_rag.predict(question)
    if match_id:
        context = patho_rag.fetch_article_context(match_id)
        return context if context != "Article context not found." else "No relevant article found."
    else:
        return "No relevant article found."

# def generate_response(question):
#     context = get_context_and_question(question)
#     response = llm_chain.run({"context": context, "question": question}).strip()  # Ensure response is properly formatted
    
#     # Extract the title and chunks for references
#     if "Article Title:" in context and "Article Chunks:" in context:
#         article_title, article_chunks = context.split("\nArticle Chunks:\n", 1)
#         references = f"Article Title: {article_title.strip('Article Title:')}\nArticle Chunks:\n{article_chunks.strip()}"
#     else:
#         references = "No references available as the provided context did not match the expected format."

#     # Format and display the output
#     formatted_output = f"Question:\n{question}\n\nSolution:\n{response}\n\nReferences:\n{references}"
#     return formatted_output

def generate_response(question):
    context = get_context_and_question(question)
    llm_result = llm_chain.run({"context": context, "question": question})

    # Debug print to check the raw result
    print(f"LLM Result Raw: {llm_result}")

    # Assuming llm_result is a string
    response_text = llm_result.strip() if isinstance(llm_result, str) else "No response generated."

    # Debug print to check the processed response
    print(f"Processed Response: {response_text}")

    if response_text != "I don't know." and "Article Title:" in context:
        article_title, article_chunks = context.split("\nArticle Chunks:\n", 1)
        article_title = article_title.replace("Article Title:", "").strip()
        references = f"Article Title: {article_title}\nArticle Chunks:\n{article_chunks.strip()}"
        formatted_output = f"Question:\n{question}\n\nSolution:\n{response_text}\n\nReferences:\n{references}"
    else:
        formatted_output = f"Question:\n{question}\n\nSolution:\n{response_text}\n\nReferences:\nNo references available as the answer could not be determined or no response was generated."

    return formatted_output


# Example usage
if __name__ == "__main__":
    question = "Which type of cancer is associated with a mutation in the HER-2 gene?"
    print(generate_response(question))
