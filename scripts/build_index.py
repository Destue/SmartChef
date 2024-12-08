# build_index.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_search_index(input_path: str, output_dir: str, batch_size: int = 32):
    """
    Build a recipe search index
    
    Args:
        input_path (str): Path to the processed recipe data
        output_dir (str): Output directory to save embeddings and index
        batch_size (int): Batch size to control memory usage
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load processed data
        logger.info(f"Loading processed data from: {input_path}")
        processed_recipes = pd.read_csv(input_path)
        
        # Initialize the model
        logger.info("Loading Sentence Transformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate embeddings
        logger.info("Starting to generate embeddings...")
        search_texts = processed_recipes['search_text'].tolist()
        
        # Generate embeddings in batches to save memory
        embeddings_list = []
        for i in range(0, len(search_texts), batch_size):
            batch_texts = search_texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
            logger.info(f"Processed {i + len(batch_texts)}/{len(search_texts)} records")
        
        # Combine all embeddings
        embeddings = np.vstack(embeddings_list)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Save embeddings
        embeddings_path = output_dir / "recipe_embeddings.npy"
        logger.info(f"Saving embeddings to: {embeddings_path}")
        np.save(embeddings_path, embeddings)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index
        index_path = output_dir / "recipe_index.faiss"
        logger.info(f"Saving FAISS index to: {index_path}")
        faiss.write_index(index, str(index_path))
        
        # Save the complete data with embeddings
        output_data_path = output_dir / "final_recipes_with_embeddings.csv"
        logger.info(f"Saving final data to: {output_data_path}")
        processed_recipes.to_csv(output_data_path, index=False)
        
        logger.info(f"Index construction completed! Total vectors: {index.ntotal}")
        
        return embeddings, index
        
    except Exception as e:
        logger.error(f"Error while building index: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Build Recipe Search Index')
    parser.add_argument('--input', default='data/processed_raw_recipes.csv',
                      help='Input file path (default: data/processed_raw_recipes.csv)')
    parser.add_argument('--output-dir', default='data',
                      help='Output directory path (default: data)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size (default: 32)')

    args = parser.parse_args()

    try:
        build_search_index(args.input, args.output_dir, args.batch_size)
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
