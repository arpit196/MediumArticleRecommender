An end-to-end recommendation pipeline designed to surface relevant Medium articles for users. By decoupling the recommendation process into a Candidate Retrieval Stage (high recall) and a Learning-to-Rank Stage (high precision), the system efficiently scales to large document corpuses while maintaining personalized accuracy.
* Candidate Retrieval stage: Where all the documents are processed via an LLM to produce document embeddings along with the user's historical reads and interests to produce a user's embedding.
   The user's embedding are then matched with all the documents' embeddings via cosine similarity, and the top 100 (or top N, where N can be any no upto number of documents) articles are then
   fetched as potential candidates for recommendation.
* Ranking stage: After the retrieval stage fetches the top 100 most relevant articles, the features of these documents along with user's features are passed to a ranking model which is essentially
   an ML model (here an XGBoost model) to narrow down the list and recommend top 15 articles to the user.

🚀 Key Features & Engineering Highlights
  - Hybrid search algorithm: Combines semantic dense retrieval (LLM vector embeddings) with lexical sparse retrieval (BM25) to catch both conceptual meaning and exact keyword matches.
  - Reciprocal Rank Fusion (RRF): Implements a robust RRF algorithm to holistically merge and score candidates from disparate retrieval streams without requiring score normalization.
  - Learning-to-Rank (LTR): Utilizes an XGBoost Ranker trained specifically on Listwise ranking objectives to optimize user satisfaction.
  - Advanced Evaluation: Built-in evaluation tracking leveraging Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR).

Tech Stack & Tools
   - Core ML/Ranking: XGBoost (LambdaMART implementation)
   - Embeddings & NLP: HuggingFace Transformers / SentenceTransformers, Rank-BM25
   - Data Processing: Pandas, NumPy, Scikit-Learn
   - Vector Operations: SciPy (Cosine Similarity metrics)
