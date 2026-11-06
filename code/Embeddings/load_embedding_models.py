# Embeddings/load_embedding_models.py
import json
from sentence_transformers import SentenceTransformer
import torch
#from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer,AutoModelForMaskedLM

def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def initialize_embedding_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dense_model = SentenceTransformer('msmarco-bert-base-dot-v5', device=device)
    
    model_id = 'naver/splade-cocondenser-ensembledistil'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    model.eval()  # Set the model to evaluation mode
    
    return dense_model, model, tokenizer, device
