# preprocess.py
import pandas as pd
from ast import literal_eval
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_raw_data(input_path: str, output_path: str):
    """
    deals with the raw data and saves the processed data to a new file.
    
    Args:
        input_path (str): input(RAW_recipes.csv)
        output_path (str): output(processed_raw_recipes.csv)
    """
    try:
        logger.info(f"begin processing data from: {input_path}")
        raw_recipes = pd.read_csv(input_path)
        
        #select columns
        processed_recipes = raw_recipes[['id', 'name', 'ingredients', 'steps']].copy()
        logger.info(f"selected {len(processed_recipes)} recipes")

        #deal with the ingredients list
        logger.info("processing ingredients...")
        processed_recipes['ingredients_text'] = processed_recipes['ingredients'].apply(
            lambda x: ", ".join(literal_eval(x)) if isinstance(x, str) else ""
        )

        #deal with the steps list
        logger.info("processing steps...")
        processed_recipes['search_text'] = (
            processed_recipes['name'] + " | " + 
            processed_recipes['ingredients_text'] + " | " + 
            processed_recipes['steps']
        )

        #make sure the output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        #save the processed data
        logger.info(f"saving processed data to: {output_path}")
        processed_recipes.to_csv(output_path, index=False)
        logger.info("data processing complete")
        
        return processed_recipes

    except FileNotFoundError:
        logger.error(f"file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"an error occurred: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Process raw recipe data")
    parser.add_argument('--input', default='data/RAW_recipes.csv',
                      help='input file path (default: data/RAW_recipes.csv)')
    parser.add_argument('--output', default='data/processed_raw_recipes.csv',
                      help='output file path (default: data/processed_raw_recipes.csv)')

    args = parser.parse_args()

    try:
        process_raw_data(args.input, args.output)
    except Exception as e:
        logger.error(f"error during processing: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()