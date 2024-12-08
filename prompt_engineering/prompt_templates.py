# prompt_engineering/prompt_templates.py
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class ConversationContext:
    """Manages conversation context and user preferences"""
    user_preferences: Dict = None
    dietary_restrictions: List[str] = None
    cooking_skill_level: str = "intermediate"
    previous_queries: List[str] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []
        if self.previous_queries is None:
            self.previous_queries = []
    
    def build_prompt(self, query: str, recipes: List[Dict]) -> str:
        """Build prompt for generation"""
        recipes_text = "\n\n".join([
            f"Recipe: {r['name']}\n"
            f"Ingredients: {r['ingredients']}\n"
            f"Steps: {r['steps']}"
            for r in recipes
        ])
        
        system_prompt = """You are a professional chef assistant. Convert the recipe into this EXACT format, replacing the text in [brackets]. DO NOT include the brackets in your response.

RECIPE NAME:
[Recipe name in capitals]

INGREDIENTS:
• [ingredient 1]
• [ingredient 2]
• [ingredient 3]
• [ingredient 4]
• [ingredient 5]

COOKING STEPS:
[Write all steps as one flowing paragraph, using connecting words like 'then', 'next', and 'finally'. Make it read naturally.]"""

        prompt = f"""{system_prompt}

Based on these recipes:
{recipes_text}

Create a recipe for: {query}"""
        
        return prompt