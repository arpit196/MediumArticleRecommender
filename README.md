A two-staged model for recommending relevant Medium articles to the respective user. This model uses a
i) Retrieval stage: Where all the documents are processed via an LLM to produce document embeddings along with the user's historical reads as well as interests to produce a user's embedding.
   The user's embedding are them matched with all the documents' embeddings via cosine similarity, and the top 100 (or top N, where N can be any no upto number of documents) articles are then
   fetched as potential candidates for recommendation.
2) Ranking stage: After the retrieval stage fetches the top 100 most relevant articles, the features of these documents along with user's features are passed to a ranking model which is essentially
   an ML model (here an XGBoost model) to narrow down the list and recommend top 15 articles to the user.

Key features:
  i) Hybrid search algorithm involving vector cosine similarity and BM25.
  ii) Reciprocal Rank Fusion for holistically combining the outputs of vector embedding similarity with BM25.
  iii) Use of normalized discounted cumulative gain as an objective function to train the XGBoost model to properly rank documents in the ranking stage.
