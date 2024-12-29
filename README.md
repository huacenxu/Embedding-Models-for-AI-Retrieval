### Embedding-Models-for-AI-Retrieval

#### Overview
This project is inspired by a 2024 data science competition focused on improving document retrieval performance through advanced embedding models. Participants developed domain-specific embedding models to enhance retrieval accuracy for AI-powered search and retrieval systems. The competition emphasized creativity, collaboration, and innovation in building cutting-edge AI solutions.

#### Key Features
Custom Embedding Model:
Implements a fine-tuned embedding model using SentenceTransformers and open-source models.
Designed to optimize document retrieval accuracy in real-world scenarios.

Vector Search with FAISS:
Utilizes FAISS for efficient indexing and similarity-based document retrieval.

Evaluation Metrics:
Evaluates models using Mean Reciprocal Rank (MRR@5) to measure relevance and accuracy of retrieved results.

Data Augmentation:
Incorporates synthetic queries to expand training datasets and improve model robustness.

#### Workflow
Data Preparation:
Processes a corpus of documents (chunked) and query datasets for training and evaluation.

Model Training:
Fine-tunes pre-trained embedding models with advanced loss functions like MultipleNegativesSymmetricRankingLoss.

Document Indexing:
Encodes and indexes documents using FAISS for efficient search.

Query Retrieval:
Searches indexed documents using query embeddings.

Evaluation:
Measures retrieval accuracy with MRR@5 and reports metrics.

#### Key Takeaways
Our script placed us among the top 5 teams for MRR@5 out of approximately 50 participating teams.

Insights from the Top 3 Teams
Top Techniques:
1. Data augmentation with synthetic queries to expand the training dataset.
2. Experimenting with advanced loss functions and hyperparameter tuning.
3. Adding linear adapters and low-rank adaptation (LoRA).
4. Leveraging ensemble methods and fine-tuning retriever/re-ranker models.

Challenges:
1. Limited compute resources.
2. Time constraints for experimentation.

Future Scope:
1. Exploring advanced pre-processing techniques.
2. Utilizing masked language modeling.
3. Improving document chunking methods.


#### Usage
Clone Repository:
git clone https://github.com/yourusername/embedding-model-competition.git

cd embedding-model-competition

Prepare Data: 
Place your data in the data/ directory. The data should include two columns: query and positive, representing questions and corresponding answers. Ensure the following files are included:
document_corpus.csv for documents (original data).
train_queries.csv and test_queries.csv for training and testing queries.

Run the code:
Train and evaluate the model:
python src/run_pipeline.py

Output:
The script generates metrics and evaluation reports in the results/ directory.
Example Metrics
Metric	Train MRR@5	Test MRR@5
Baseline	0.713	0.762
Fine-tuned	0.790	0.809

#### Requirements
Python 3.8+
Required libraries: torch, transformers, faiss, sentence-transformers, datasets, numpy.
