from transformers import pipeline
from keybert import KeyBERT

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Initialize KeyBERT for keyword extraction
keyword_extractor = KeyBERT()
