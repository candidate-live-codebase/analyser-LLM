import pandas as pd
from collections import Counter
from transformers import pipeline
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

# Initialize the sentiment pipeline and keyword extractor
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0
)
keyword_extractor = KeyBERT()

# Function to remove stop words
def remove_stop_words(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in all_stopwords]
    return " ".join(filtered_tokens)

# Function to process each tweet
def process_tweet(tweet_content):
    tweet_content_clean = remove_stop_words(tweet_content)
    sentiment_result = sentiment_pipeline(tweet_content_clean)
    label = sentiment_result[0]["label"]
    score = sentiment_result[0]["score"]

    normalized_score = -score if label == "NEGATIVE" else 1 + score

    keywords = keyword_extractor.extract_keywords(tweet_content_clean, top_n=5)
    trending_keywords = [kw[0] for kw in keywords]

    return {
        "content_sentiment": {
            "label": label,
            "confidence_score": score,
            "normalized_score": normalized_score,
        },
        "trending_keywords": trending_keywords
    }

# Main processing function
def process_data(data):
    processed_posts = []
    tweet_texts = []

    for _, row in data.iterrows():
        tweet_content = row["content"]
        likes = row.get("likes", 0)
        shares = row.get("shares", 0)
        views = row.get("views", 0)
        comments = row.get("reply_count", 0)
        engagement_score = likes + shares + comments + views

        processed_data = process_tweet(tweet_content)
        processed_posts.append({
            "tweet_id": row["tweet_id"],
            "url": row.get("url", ""),
            "content_sentiment": processed_data["content_sentiment"],
            "trending_keywords": processed_data["trending_keywords"],
            "engagement_score": engagement_score,
            "likes": likes,
            "shares": shares,
            "timestamp": row.get("datetime"),
            "comments": row.get("reply_count")
        })
        tweet_texts.append(remove_stop_words(tweet_content))

    processed_posts_df = pd.DataFrame(processed_posts)
    processed_posts_df['timestamp'] = pd.to_datetime(processed_posts_df['timestamp'])
    processed_posts_df['normalized_score'] = processed_posts_df['content_sentiment'].apply(lambda x: x['normalized_score'])

    min_engagement = processed_posts_df['engagement_score'].min()
    max_engagement = processed_posts_df['engagement_score'].max()

    processed_posts_df['normalized_engagement_score'] = (
        (processed_posts_df['engagement_score'] - min_engagement) /
        (max_engagement - min_engagement) * 100
    ) if max_engagement != min_engagement else 0

    top_5_posts = processed_posts_df.nlargest(5, 'engagement_score')[['url', 'likes', 'shares', 'content_sentiment']]

    top_5_data = [{
        "url": post["url"],
        "likes": post["likes"],
        "shares": post["shares"],
        "sentiment_score": post["content_sentiment"]['normalized_score'],
        "sentiment": post["content_sentiment"]['label']
    } for _, post in top_5_posts.iterrows()]

    return generate_output(processed_posts_df, top_5_data)

# Function to generate final output
def generate_output(processed_posts_df, top_5_data):
    average_likes_per_post = processed_posts_df['likes'].mean()
    average_comments_per_post = processed_posts_df['comments'].mean()
    
    all_keywords = [kw for sublist in processed_posts_df['trending_keywords'] for kw in sublist]
    keyword_counts = Counter(all_keywords)
    top_7_keywords = keyword_counts.most_common(7)

    processed_posts_df['date'] = processed_posts_df['timestamp'].dt.date
    daily_sentiment = processed_posts_df.groupby('date')['normalized_score'].mean()

    sentiment_coordinates = [{"date": str(date), "sentiment": score} for date, score in daily_sentiment.items()]
    overall_sentiment_score = daily_sentiment.mean()

    overall_sentiment_label = "Positive" if overall_sentiment_score > 1 else ("Negative" if overall_sentiment_score < 0 else "Neutral")

    sentiment_counts = processed_posts_df['content_sentiment'].apply(lambda x: x['label']).value_counts().to_dict()
    total_positive_posts = sentiment_counts.get("POSITIVE", 0)
    total_negative_posts = sentiment_counts.get("NEGATIVE", 0)
    total_neutral_posts = len(processed_posts_df) - total_positive_posts - total_negative_posts

    return {
        "average_likes_per_post": int(average_likes_per_post),
        "average_comments_per_post": int(average_comments_per_post),
        "top_7_keywords": [{"keyword": kw[0], "count": int(kw[1])} for kw in top_7_keywords],
        "sentiment_coordinates": sentiment_coordinates,
        "overall_normalized_sentiment_score": overall_sentiment_score,
        "overall_sentiment_label": overall_sentiment_label,
        "total_positive_posts": total_positive_posts,
        "total_negative_posts": total_negative_posts,
        "total_neutral_posts": total_neutral_posts,
        "normalized_engagement_scores": [{"tweet_id": post["tweet_id"], "normalized_engagement_score": post["normalized_engagement_score"]} for _, post in processed_posts_df.iterrows()],
        "top_5_posts": top_5_data
    }
