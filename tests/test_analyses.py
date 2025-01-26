import pytest
import pandas as pd
from llm.services.analysis import process_tweet, process_data, generate_sentiment_plot

# Sample data for testing
TEST_DATA = [
    {"tweet_id": 1, "content": "This is an amazing day!", "datetime": "2023-01-01T10:00:00"},
    {"tweet_id": 2, "content": "I am feeling terrible today.", "datetime": "2023-01-01T15:00:00"},
    {"tweet_id": 3, "content": "FastAPI is awesome for web apps.", "datetime": "2023-01-02T12:00:00"}
]

@pytest.fixture
def sample_dataframe():
    """Fixture for creating a sample DataFrame."""
    return pd.DataFrame(TEST_DATA)

def test_process_tweet():
    """Test the process_tweet function."""
    tweet_content = "I love using Python!"
    result = process_tweet(tweet_content)
    
    assert "content_sentiment" in result
    assert "trending_keywords" in result
    assert isinstance(result["content_sentiment"], dict)
    assert isinstance(result["trending_keywords"], list)
    assert len(result["trending_keywords"]) > 0

def test_process_data(sample_dataframe):
    """Test the process_data function with sample data."""
    result = process_data(sample_dataframe)
    
    assert "processed_posts" in result
    assert isinstance(result["processed_posts"], list)
    assert len(result["processed_posts"]) == len(TEST_DATA)
    
    first_post = result["processed_posts"][0]
    assert "tweet_id" in first_post
    assert "content_sentiment" in first_post
    assert "trending_keywords" in first_post

def test_generate_sentiment_plot(sample_dataframe):
    """Test the generate_sentiment_plot function with sample data."""
    # This test ensures the function generates a valid base64 image string
    image_base64 = generate_sentiment_plot(sample_dataframe)
    
    assert isinstance(image_base64, str)
    assert len(image_base64) > 0
    assert image_base64.startswith("iVBOR") or image_base64.startswith("/")  # Base64 for PNG images

