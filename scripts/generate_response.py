# generate_response.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_engineering.prompt_templates import ConversationContext
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch

class RecipeResponseGenerator:
    def __init__(self, 
                 search_model_name="all-MiniLM-L6-v2",
                 gen_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 csv_path="data/final_recipes_with_embeddings.csv",
                 embedding_path="data/recipe_embeddings.npy",
                 index_path="data/recipe_index.faiss"):
        # Initialize search components
        self.search_model = SentenceTransformer(search_model_name)
        self.recipes_df = pd.read_csv(csv_path)
        self.embeddings = np.load(embedding_path)
        self.index = faiss.read_index(index_path)
        
        # Initialize generation model
        print("Loading generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Initialize conversation context
        self.context = ConversationContext()
    
    def search_recipes(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant recipes"""
        query_vector = self.search_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            recipe = self.recipes_df.iloc[idx]
            results.append({
                'name': recipe['name'],
                'ingredients': recipe['ingredients'],
                'steps': recipe['steps']
            })
        return results

    def generate_response(self, prompt: str) -> str:
        """Generate response using the language model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
            
            outputs = self.gen_model.generate(
                **inputs,
                max_new_tokens=600,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def process_query(self, 
                     query: str, 
                     user_preferences: Dict = None,
                     dietary_restrictions: List[str] = None,
                     skill_level: str = None) -> str:
        """Process user query and generate response"""
        # Update context if provided
        if user_preferences:
            self.context.user_preferences.update(user_preferences)
        if dietary_restrictions:
            self.context.dietary_restrictions.extend(dietary_restrictions)
        if skill_level:
            self.context.cooking_skill_level = skill_level
            
        # Add query to history
        if self.context.previous_queries is None:
            self.context.previous_queries = []
        self.context.previous_queries.append(query)
        
        # Search and generate
        relevant_recipes = self.search_recipes(query)
        prompt = self.context.build_prompt(query, relevant_recipes)
        return self.generate_response(prompt)

def main():
    print("Initializing recipe response generator...")
    generator = RecipeResponseGenerator()
    
    # Set initial user parameters
    user_prefs = {"spice_level": "medium"}
    diet_restrictions = ["vegetarian"]
    skill = "beginner"
    
    while True:
        query = input("\nEnter your recipe query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        print("\nProcessing your query...")
        response = generator.process_query(
            query,
            user_preferences=user_prefs,
            dietary_restrictions=diet_restrictions,
            skill_level=skill
        )
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()