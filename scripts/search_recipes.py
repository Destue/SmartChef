import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class RecipeSearcher:
    def __init__(self, 
                 model_name="all-MiniLM-L6-v2",
                 csv_path="data/final_recipes_with_embeddings.csv",
                 embedding_path="data/recipe_embeddings.npy",
                 index_path="data/recipe_index.faiss"):
        """
        Initialize the RecipeSearcher with model and data paths.

        Args:
            model_name (str): Name of the Sentence Transformer model to load.
            csv_path (str): Path to the CSV file containing recipes with embeddings.
            embedding_path (str): Path to the NumPy file containing recipe embeddings.
            index_path (str): Path to the FAISS index file.
        """
        # Load the model and data
        self.model = SentenceTransformer(model_name)
        self.recipes_df = pd.read_csv(csv_path)
        self.embeddings = np.load(embedding_path)
        self.index = faiss.read_index(index_path)
        
        print(f"Loaded {len(self.recipes_df)} recipes")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        
    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Search for the top k most similar recipes to the query.

        Args:
            query (str): The search query string.
            k (int): The number of top similar recipes to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the search results with recipe details and similarity scores.
        """
        # Convert the query to a vector
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # Use FAISS to search for the most similar vectors
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        # Retrieve the corresponding recipe information
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.recipes_df):
                recipe = self.recipes_df.iloc[idx]
                results.append({
                    'id': recipe['id'],
                    'name': recipe['name'],
                    'ingredients': recipe['ingredients'],
                    'similarity_score': 1 / (1 + distance)  # Convert distance to similarity score
                })
            else:
                # Handle cases where the index might be out of bounds
                results.append({
                    'id': None,
                    'name': None,
                    'ingredients': None,
                    'similarity_score': None
                })
        
        return pd.DataFrame(results)

# Usage Example
if __name__ == "__main__":
    searcher = RecipeSearcher()
    
    # Test search
    query = "chicken soup"
    results = searcher.search(query)
    print(f"\nSearch results for '{query}':")
    print(results[['name', 'similarity_score']])
