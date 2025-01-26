import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from llm.services.utils import remove_stop_words
from llm.models.sentiment import sentiment_pipeline, keyword_extractor

def process_tweet(tweet_content):
    tweet_content_clean = remove_stop_words(tweet_content)
    sentiment_result = sentiment_pipeline(tweet_content_clean)
    keywords = keyword_extractor.extract_keywords(tweet_content_clean, top_n=5)
    return {
        "content_sentiment": {
            "label": sentiment_result[0]["label"],
            "score": sentiment_result[0]["score"]
        },
        "trending_keywords": [kw[0] for kw in keywords]
    }

def process_data(data):
    processed_posts = []
    tweet_texts = []
    for _, row in data.iterrows():
        tweet_content = row["content"]
        processed_data = process_tweet(tweet_content)
        processed_posts.append({
            "tweet_id": row["tweet_id"],
            "content_sentiment": processed_data["content_sentiment"],
            "trending_keywords": processed_data["trending_keywords"],
        })
        tweet_texts.append(remove_stop_words(tweet_content))
    # Further analysis (e.g., engagement, sentiment aggregation)...
    # Return structured analysis results
    return {
        # Placeholder for analysis results
        "processed_posts": processed_posts
    }

def generate_sentiment_plot(data):
    # Plot daily sentiment analysis
    daily_sentiment = pd.DataFrame(process_data(data)["daily_sentiment"])
    plt.figure(figsize=(10, 6))
    daily_sentiment.plot(kind="line")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.read()).decode("utf-8")
