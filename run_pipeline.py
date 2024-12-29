import torch
import faiss
import json
import logging
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

def index_documents(model, documents, batch_size=32):
    embeddings = model.encode(documents, batch_size=batch_size).cpu().numpy()
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search_index(model, queries, index, batch_size=32, top_k=5):
    query_embeddings = model.encode(queries, batch_size=batch_size).cpu().numpy()
    faiss.normalize_L2(query_embeddings)
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices

def evaluate(predictions, ground_truth, top_k=5):
    mrr = 0
    for i, truth in enumerate(ground_truth):
        if truth in predictions[i][:top_k]:
            rank = predictions[i][:top_k].tolist().index(truth) + 1
            mrr += 1 / rank
    return mrr / len(ground_truth)

def run_pipeline(data_dir='data', output_dir='results'):
    model = EmbeddingModel()
    
    logger.info("Loading data...")
    documents = load_dataset('csv', data_files=str(Path(data_dir) / 'document_corpus.csv'), split='train')['text']
    queries = load_dataset('csv', data_files=str(Path(data_dir) / 'test_queries.csv'), split='train')
    
    logger.info("Indexing documents...")
    index = index_documents(model, documents)

    logger.info("Searching queries...")
    predictions, _ = search_index(model, queries['text'], index)

    logger.info("Evaluating results...")
    mrr = evaluate(predictions, queries['ground_truth'])
    logger.info(f"MRR@5: {mrr:.4f}")

    Path(output_dir).mkdir(exist_ok=True)
    with open(Path(output_dir) / 'metrics.json', 'w') as f:
        json.dump({"MRR@5": mrr}, f, indent=4)

if __name__ == "__main__":
    run_pipeline()
