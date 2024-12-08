# fine_tuning/scripts/prepare_data.py
import pandas as pd
import json
from typing import List, Dict
import random
from sklearn.model_selection import train_test_split

def format_recipe(row: pd.Series) -> Dict:
    """Format a single recipe into training data format"""
    input_prompt = f"Create a recipe for: {row['name']}"
    
    target_output = f"""RECIPE NAME:
{row['name'].upper()}

INGREDIENTS:
{format_ingredients(eval(row['ingredients']))}

COOKING STEPS:
{format_steps(eval(row['steps']))}"""
    
    return {
        "input": input_prompt,
        "output": target_output
    }

def format_ingredients(ingredients: List[str]) -> str:
    """Format ingredients list into bullet points"""
    return '\n'.join(f"• {ingredient}" for ingredient in ingredients[:6])

def format_steps(steps: List[str]) -> str:
    """Format cooking steps into a flowing paragraph"""
    connecting_words = ['Then', 'Next', 'After that', 'Finally']
    formatted_steps = []
    
    for i, step in enumerate(steps):
        if i == 0:
            formatted_steps.append(step.capitalize())
        else:
            formatted_steps.append(f"{random.choice(connecting_words)}, {step.lower()}")
    
    return ' '.join(formatted_steps)

def prepare_training_data(input_path: str, output_dir: str, sample_size: int = 500, test_size: float = 0.1):
    """
    Prepare training and validation datasets
    
    Args:
        input_path: Path to raw data file
        output_dir: Directory to save processed data
        sample_size: Number of recipes to sample (increased to 500)
        test_size: Proportion of data to use for validation
    """
    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    print(f"Sampling {sample_size} recipes...")
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print("Formatting recipes...")
    formatted_data = []
    for _, row in sampled_df.iterrows():
        try:
            formatted_data.append(format_recipe(row))
        except Exception as e:
            print(f"Error processing recipe {row['name']}: {e}")
            continue
    
    print("Splitting into train and validation sets...")
    train_data, val_data = train_test_split(
        formatted_data, 
        test_size=test_size, 
        random_state=42
    )
    
    print("Saving processed data...")
    # Save training data
    with open(f"{output_dir}/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation data
    with open(f"{output_dir}/val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Successfully saved {len(train_data)} training examples and {len(val_data)} validation examples")

if __name__ == "__main__":
    # Configuration
    input_path = "data/RAW_recipes.csv"    # Path to raw data
    output_dir = "fine_tuning/data"        # Output directory
    sample_size = 500                      # Increased from 100 to 500
    
    prepare_training_data(input_path, output_dir, sample_size=sample_size)