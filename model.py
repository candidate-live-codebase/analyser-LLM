import pandas as pd
import json
import re
import openai  # Ensure OpenAI library is installed and configured
import pandas as pd
from collections import Counter
from transformers import pipeline
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from dotenv import load_dotenv
import openai
import os

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more details
logger = logging.getLogger(__name__)

# Example of replacing print with logging

# load_dotenv()  # Load the .env file
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if openai.api_key is None:
#     raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

import os
os.environ["OPENAI_API_KEY"] = "your api key"

load_dotenv()  # Load the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError

# Hindi stopwords (common list, can be expanded)
hindi_stopwords = set([
    'और', 'की', 'के', 'में', 'है', 'हैं', 'से', 'को', 'था', 'थी', 'यह', 'वह', 'तुम', 'मैं', 'उनका', 'उसका', 'रहा', 'रही',
    'कहा', 'कहीं', 'यहां', 'वहां', 'तभी', 'बड़ा', 'कुछ', 'इतना', 'सभी', 'ही', 'अभी', 'आदि', 'नहीं', 'तुम्हारा', 'आपका',
    'अपने', 'इन', 'उन', 'उनकी', 'क्योंकि', 'यदि', 'कर', 'करने', 'पर', 'अच्छा', 'बुरा', 'गया', 'ने'
])

# Hinglish stopwords (commonly used transliterations, can be extended)
hinglish_stopwords = set([
    'aur', 'ki', 'ke', 'mein', 'hai', 'hain', 'se', 'ko', 'tha', 'thi', 'yaha', 'waha', 'abhi', 'nahi', 'kya', 'kyu',
    'apna', 'tera', 'mera', 'unka', 'unka', 'uska', 'kyunki', 'agar', 'acha', 'bura', 'gaya', 'kar', 'karna'
])

# Merge all stopwords
all_stopwords = ENGLISH_STOP_WORDS.union(hindi_stopwords).union(hinglish_stopwords)

keyword_extractor = KeyBERT()

# Function to remove stop words
def remove_stop_words(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in all_stopwords]
    return " ".join(filtered_tokens)

# Function to process each tweet
def process_tweet(tweet_content):
    tweet_content_clean = remove_stop_words(tweet_content)
  

    keywords = keyword_extractor.extract_keywords(tweet_content_clean, top_n=5)
    trending_keywords = [kw[0] for kw in keywords]

    return trending_keywords

def analyze_content(client, content):
    messages = [
        {"role": "system", "content": "You are an AI that strictly returns JSON-formatted responses. Do not include explanations or extra text."},
        {"role": "user", "content": f"""
        Analyze the following content and provide:
        - The main category (e.g., Technology, Healthcare, Finance).
        - Sentiment classification (Positive, Neutral, Negative).
        - Sentiment score (-1 to 1, where -1 is very negative, 0 to 1 is neutral, 1 to 2 is positive) , Note precsion upto 2 decimal point.

        Return JSON in this exact format:
        {{
            "category": "CategoryName",
            "sentiment": "Positive/Neutral/Negative",
            "sentiment_score": 0.XX
        }}

        Content:
        \"\"\"{content}\"\"\"  
        """}
    ]

    try:
        response = client.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.0
        )

        response_text = response["choices"][0]["message"]["content"].strip()
        
        # Remove ```json ... ``` formatting if present
        response_text = re.sub(r"```json\s*", "", response_text)  
        response_text = re.sub(r"```$", "", response_text) 
        
        return json.loads(response_text)
        
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from response. Raw response: {response_text}")
        return {"category": "Unknown", "sentiment": "Neutral", "sentiment_score": 0.0}
    except Exception as e:
        print(f"Error analyzing content: {e}")
        return {"category": "Unknown", "sentiment": "Neutral", "sentiment_score": 0.0}

def process_dataframe(df, client, batch_size=1):
    results = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]

        batch_results = batch['content'].apply(lambda x: analyze_content(client, x) if pd.notna(x) else None)
        results.extend(batch_results)

    return {
        "content_sentiment": {
            "label": [res.get('sentiment', 'Neutral') for res in results],
            "confidence_score": [res.get('sentiment_score', 0.0) for res in results],
            "category": [res.get('category', 'Unknown') for res in results]
        },
        "trending_keywords": process_tweet(df.iloc[0]['content'])

    }

def process_data(data):
    processed_posts = []

    for _, row in data.iterrows():
        if row["tweet_id"] == 3529525859168661060:
            logger.info(f"row_engagement __________________ {row}")

        tweet_content = row["content"]
        
        # Ensure numeric values, replace NaN with 0
        likes = pd.to_numeric(row.get("likes", 0), errors="coerce") or 0
        shares = pd.to_numeric(row.get("shares", 0), errors="coerce") or 0
        views = pd.to_numeric(row.get("views", 0), errors="coerce") or 0
        comments = pd.to_numeric(row.get("reply_count", 0), errors="coerce") or 0

        engagement_score = likes + shares + comments + views

        # Check if engagement is still null
        if pd.isna(engagement_score):
            engagement_score = 0  # Set to 0 if still NaN
        
        client = openai
        processed_data = process_dataframe(pd.DataFrame([{"content": tweet_content}]), client)

        processed_posts.append({
            "tweet_id": row["tweet_id"],
            "url": row.get("url", ""),
            "content_sentiment": processed_data["content_sentiment"],
            "trending_keywords": processed_data["trending_keywords"],
            "engagement_score": engagement_score,
            "likes": likes,
            "shares": shares,
            "timestamp": row.get("datetime"),
            "comments": comments
        })

    processed_posts_df = pd.DataFrame(processed_posts)


    # ✅ Convert timestamp to datetime format
    processed_posts_df['timestamp'] = pd.to_datetime(
        processed_posts_df['timestamp'], 
        errors='coerce'
    )
    min_engagement = processed_posts_df['engagement_score'].min()
    max_engagement = processed_posts_df['engagement_score'].max()

    processed_posts_df['normalized_engagement_score'] = (
        (processed_posts_df['engagement_score'] - min_engagement) /
        (max_engagement - min_engagement) * 100
    ) if max_engagement != min_engagement else 0

    # ✅ Extract only the date
    processed_posts_df['date'] = processed_posts_df['timestamp'].dt.date

    # ✅ Extract Sentiment Score before calculating mean
    processed_posts_df['sentiment_score'] = processed_posts_df['content_sentiment'].apply(
        lambda x: x['confidence_score'][0] if isinstance(x, dict) and 'confidence_score' in x else 0
    )

    daily_sentiment = processed_posts_df.groupby('date')['sentiment_score'].mean()
    
    # ✅ Correct Overall Sentiment Score Calculation
    overall_sentiment_score = processed_posts_df['sentiment_score'].mean()

    overall_sentiment_label = "Positive" if overall_sentiment_score > 1 else (
        "Negative" if overall_sentiment_score < 0 else "Neutral"
    )

    sentiment_counts = processed_posts_df['content_sentiment'].apply(lambda x: x['label'][0]).value_counts().to_dict()
    total_positive_posts = sentiment_counts.get("Positive", 0)
    total_negative_posts = sentiment_counts.get("Negative", 0)
    total_neutral_posts = len(processed_posts_df) - total_positive_posts - total_negative_posts
    
    all_keywords = [kw for sublist in processed_posts_df['trending_keywords'] for kw in sublist]
    keyword_counts = Counter(all_keywords)
    top_7_keywords = keyword_counts.most_common(7)

    top_5_posts = processed_posts_df.nlargest(5, 'engagement_score')[['url', 'likes', 'shares', 'content_sentiment']]

    top_5_data = [{
        "url": post["url"],
        "likes": post["likes"],
        "shares": post["shares"],
        "sentiment_score": post["content_sentiment"]['confidence_score'],
        "sentiment": post["content_sentiment"]['label'],
        "category":post["content_sentiment"]['category']
    } for _, post in top_5_posts.iterrows()]

    return {
        "average_likes_per_post": int(processed_posts_df['likes'].mean() or 0),
        "average_comments_per_post": int(processed_posts_df['comments'].mean() or 0),
        "top_7_keywords": [{"keyword": kw[0], "count": int(kw[1])} for kw in top_7_keywords],
        "sentiment_coordinates": [{"date": str(date), "sentiment": score} for date, score in daily_sentiment.items()],
        "overall_normalized_sentiment_score": overall_sentiment_score,
        "overall_sentiment_label": overall_sentiment_label,
        "total_positive_posts": total_positive_posts,
        "total_negative_posts": total_negative_posts,
        "total_neutral_posts": total_neutral_posts,
        "normalized_engagement_scores": [{"tweet_id": post["tweet_id"], "normailzed_engagement_score": post["normalized_engagement_score"]} for _, post in processed_posts_df.iterrows()],
        "top_5_posts": top_5_data
    }


