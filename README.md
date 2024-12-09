# SmartChef: AI Recipe Generation System 🧑‍🍳

SmartChef is an intelligent recipe generation system that combines RAG (Retrieval-Augmented Generation), fine-tuning, and prompt engineering to create personalized cooking recipes. The system can understand your recipe queries and generate detailed, contextually relevant cooking instructions.

## Features

- **Recipe Search**: Efficient recipe retrieval using FAISS indexing
- **Fine-tuned Generation**: Custom recipe generation using fine-tuned GPT model
- **Structured Output**: Well-formatted recipes with ingredients and cooking steps
- **Local Operation**: Runs completely locally without external API dependencies

## Project Structure
## Project Structure
```
SmartChef/
├── data/                           # Data directory
│   ├── final_recipes_with_embeddings.csv   # Processed recipes with embeddings
│   ├── processed_raw_recipes.csv           # Initial processed recipes
│   ├── RAW_recipes.csv                     # Original dataset from Kaggle
│   ├── recipe_embeddings.npy              # Neural embeddings for recipes
│   └── recipe_index.faiss                 # FAISS search index
│
├── fine_tuning/                    # Fine-tuning module
│   ├── data/
│   │   ├── train.json              # Training data
│   │   └── val.json                # Validation data
│   ├── models/
│   │   ├── recipe-bot/             # Training checkpoints
│   │   └── recipe-bot-final/       # Final model
│   └── scripts/
│       ├── prepare_data.py         # Prepare training data
│       ├── test_model.py          # Test fine-tuned model
│       └── train.py               # Training script
│
├── prompt_engineering/             # Prompt engineering module
│   ├── __init__.py
│   └── prompt_templates.py        # Prompt templates
│
├── scripts/                        # Main scripts
│   ├── build_index.py             # Build search index
│   ├── generate_response.py       # Generate recipe responses
│   ├── preprocess.py             # Data preprocessing
│   ├── present_test.py           # Testing script
│   └── search_recipes.py         # Recipe search functionality
│
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## Setup

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- [Raw Recipe Dataset](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) from Kaggle

### Installation

1. Clone the repository
```bash
git clone https://github.com/Destue/SmartChef.git
cd SmartChef
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Data Setup

1. Download the RAW_recipes.csv from Kaggle link above
2. Place it in the `data/` directory
3. Run preprocessing:
```bash
python scripts/preprocess.py
```

4. Build search index:
```bash
python scripts/build_index.py
```

### Fine-tuning (Optional)

To train your own model:
```bash
# Prepare training data
python fine_tuning/scripts/prepare_data.py

# Train model
python fine_tuning/scripts/train.py
```

## Usage

### Using RAG System
```bash
python scripts/generate_response.py
```

### Using Fine-tuned Model
```bash
python fine_tuning/scripts/test_model.py
```

## System Components

### 1. RAG System
- Uses FAISS for efficient similarity search
- Combines retrieved recipes with generation
- Provides context-aware responses

### 2. Fine-tuning
- Fine-tuned on curated recipe dataset
- Optimized for structured recipe generation
- Supports customization through training

### 3. Prompt Engineering
- Carefully designed prompts
- Structured output format
- Context management

## Example Output

```
RECIPE NAME:
CHICKEN STIR FRY

INGREDIENTS:
• Chicken breast
• Broccoli
• Garlic
• Soy sauce
• Vegetable oil

COOKING STEPS:
First, cut chicken into bite-sized pieces. Then heat oil in a large pan...
```

## Development

The system is built with modularity in mind, allowing for easy extensions and modifications:
- Add new recipe sources
- Modify generation parameters
- Customize prompts
- Implement new features

## Future Improvements

- Add support for dietary restrictions
- Improve recipe step generation
- Add more recipe metadata
- Enhance search capabilities
- Implement user feedback system

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset from [Food.com](https://www.food.com/)
- Built with Hugging Face Transformers
- Uses FAISS for vector search
