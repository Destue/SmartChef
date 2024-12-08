# fine_tuning/scripts/test_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def format_output(text: str) -> str:
    """Format and clean the generated text"""
    # Remove the input prompt if it appears in the output
    if "Create a recipe for:" in text:
        text = text.split("Create a recipe for:")[1].strip()
        
    # Remove URLs and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[@#]|\\[a-z]', '', text)
    
    # Clean up extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def generate_recipe(query: str, model_path: str = "fine_tuning/models/recipe-bot-final"):
    """Generate recipe using fine-tuned model with improved prompt"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # More specific prompt template
    input_text = f"""Create a recipe for: {query}

RECIPE NAME:
{query.upper()}

INGREDIENTS:
• {query}
• Garlic
• Onion
• Salt and pepper
• Olive oil

COOKING STEPS:
1. Prepare {query} by cutting into appropriate sizes.
2."""
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    
    print("Generating recipe...")
    outputs = model.generate(
        **inputs,
        max_length=250,           # Shorter length to avoid wandering
        min_length=100,          # Ensure some content
        num_return_sequences=1,
        temperature=0.7,         # Lower temperature for more focused output
        do_sample=True,
        repetition_penalty=1.3,  # Stronger repetition penalty
        no_repeat_ngram_size=3,
        top_p=0.85,             # More conservative sampling
        top_k=40,               # Limited vocabulary choices
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Process and format the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = format_output(generated_text)
    
    # Add section headers if they're missing
    if "INGREDIENTS:" not in formatted_text:
        sections = formatted_text.split("COOKING STEPS:")
        if len(sections) == 2:
            formatted_text = f"INGREDIENTS:\n{sections[0]}\nCOOKING STEPS:\n{sections[1]}"
    
    return formatted_text

def main():
    print("Recipe Generator (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        query = input("\nWhat would you like to cook? ")
        if query.lower() == 'quit':
            break
        
        try:
            recipe = generate_recipe(query)
            print("\nGenerated Recipe:")
            print("=" * 50)
            print(recipe)
            print("=" * 50)
        except Exception as e:
            print(f"Error generating recipe: {str(e)}")

if __name__ == "__main__":
    main()