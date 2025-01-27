# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from model import process_data

app = FastAPI()

# Define the structure of the input data using Pydantic
class Tweet(BaseModel):
    tweet_id: int
    content: str
    url: str
    likes: int
    shares: int
    views: int
    reply_count: int
    datetime: str

class ResponseItem(BaseModel):
    tweet_id: int
    url: str
    content_sentiment: Dict[str, str]
    trending_keywords: List[str]
    engagement_score: int
    likes: int
    shares: int
    timestamp: str

class OutputResponse(BaseModel):
    average_likes_per_post: float
    average_comments_per_post: int
    top_7_keywords: List[Dict[str, int]]
    sentiment_coordinates: List[Dict[str, str]]
    overall_normalized_sentiment_score: float
    overall_sentiment_label: str
    total_positive_posts: int
    total_negative_posts: int
    total_neutral_posts: int
    normalized_engagement_scores: List[Dict[str, float]]
    top_5_posts: List[Dict[str, str]]


@app.post("/process_twitter_data", response_model=OutputResponse)
def process_twitter_data(tweets: List[Tweet]):
    # Convert the list of tweets into a DataFrame
    tweet_data = [tweet.dict() for tweet in tweets]
    df = pd.DataFrame(tweet_data)

    # Process the data using the model
    output = process_data(df)

    # Return the processed output
    return output
