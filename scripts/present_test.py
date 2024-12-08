import pandas as pd
from search_recipes import search_recipes
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gc

# Load tokenizer and model once
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Load processed data
recipes = pd.read_csv("data/final_recipes_with_embeddings.csv")

def decode_tokens(tokens_column):
    """
    Decode tokenized data into natural language strings.
    """
    try:
        if not isinstance(tokens_column, str):
            return "Invalid token format."
        tokens = eval(tokens_column)
        if not isinstance(tokens, list):
            return "Invalid token format."
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded_text
    except Exception as e:
        return f"Error decoding: {e}"

def generate_response(decoded_name, decoded_ingredients, decoded_steps):
    """
    Generate a user-friendly recipe description.
    """
    prompt = (
        f"Recipe Name: {decoded_name}\n"
        f"Ingredients:\n{decoded_ingredients}\n"
        f"Steps:\n{decoded_steps}\n\n"
        f"Please rewrite this recipe into a user-friendly description with:\n"
        f"1. A concise title\n"
        f"2. A bullet-point list of ingredients\n"
        f"3. Numbered step-by-step instructions.\n"
    )
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=300,  # Limit length to avoid memory issues
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True  # Enable sampling
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating response: {e}"

def chatbot():
    """
    Main chatbot loop.
    """
    print("Welcome to SmartChef!")
    print("Enter ingredients or a query to get recipe suggestions.")
    print("Type 'exit' to quit the chatbot.\n")
    
    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        print("\nSearching for recipes...\n")
        try:
            top_matches = search_recipes(user_query, top_k=3)
            if top_matches.empty:
                print("No matching recipes found. Please try again with a different query.\n")
                continue
            
            print("Top matching recipes:\n")
            for index, row in top_matches.iterrows():
                decoded_name = decode_tokens(row["name_tokens"])
                decoded_ingredients = decode_tokens(row["ingredient_tokens"])
                decoded_steps = decode_tokens(row["steps_tokens"])
                
                print(f"Recipe {index + 1}:")
                print(f"  Recipe ID: {row['id']}")
                print(f"  Recipe Name: {decoded_name}")
                print(f"  Distance: {row['distance']:.4f}")
                print("  Generating description...\n")
                
                description = generate_response(decoded_name, decoded_ingredients, decoded_steps)
                print(f"  Description:\n{description}\n")
            
            # Clear resources
            gc.collect()

        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    chatbot()

