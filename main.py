from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
from elasticsearch import AsyncElasticsearch
from datetime import datetime
import uvicorn
from model import process_data
import logging
from datetime import datetime
import json
import base64

import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI and Elasticsearch
app = FastAPI()
es = AsyncElasticsearch(
    cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
    api_key=os.getenv('ELASTIC_API_KEY')
)

# Define our data models
class Tweet(BaseModel):
    id: str
    phone: str
    content: str
    datetime: str
    heading: str = ""  
    likes: int
    shares: int
    source: str
    username: str
    url: str  
    views: int 
    reply_count: int

class UserTweetsRequest(BaseModel):
    user_id: str
    phone: str
    tweet_count: int
    tweets: List[Tweet]

class OutputResponse(BaseModel):
    average_likes_per_post: int
    average_comments_per_post: int
    top_7_keywords: List[Dict[str, str | int]]
    sentiment_coordinates: List[Dict[str, str | float]]
    overall_normalized_sentiment_score: float
    overall_sentiment_label: str
    total_positive_posts: int
    total_negative_posts: int
    total_neutral_posts: int
    normalized_engagement_scores: List[Dict[str, str | float]]
    top_5_posts: List[Dict[str, str | int | float]]

ELASTICSEARCH_MAPPING = {
    "mappings": {
        "properties": {
            "metadata": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "phone": {"type": "keyword"},
                    "tweet_count": {"type": "integer"}
                }
            },
            "processing_timestamp": {"type": "date"},
            "tweet_analysis": {
                "properties": {
                    "average_likes_per_post": {"type": "float"},
                    "average_comments_per_post": {"type": "float"},
                    "top_7_keywords": {
                        "type": "nested",
                        "properties": {
                            "keyword": {"type": "keyword"},
                            "count": {"type": "integer"}
                        }
                    },
                    "sentiment_coordinates": {
                        "type": "nested",
                        "properties": {
                            "tweet_id": {"type": "keyword"},
                            "sentiment": {"type": "float"}
                        }
                    },
                    "overall_normalized_sentiment_score": {"type": "float"},
                    "overall_sentiment_label": {"type": "keyword"}
                }
            },
            "raw_tweets": {
                "type": "nested",
                "properties": {
                    "id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "datetime": {"type": "keyword"},  # Changed to keyword to store as string
                    "likes": {"type": "integer"},
                    "shares": {"type": "integer"},
                    "views": {"type": "integer"},
                    "url": {"type": "keyword"}
                }
            }
        }
    },
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1
    }
}

async def initialize_elasticsearch():
    """Initialize Elasticsearch index if it doesn't exist."""
    try:
        index_name = "tweets_analysis"
        exists = await es.indices.exists(index=index_name)
        if exists:
            await es.indices.delete(index=index_name)
            logger.info(f"Deleted existing index {index_name}")
        
        await es.indices.create(
            index=index_name,
            body=ELASTICSEARCH_MAPPING,
            request_timeout=30
        )
        logger.info(f"Created index {index_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch: {str(e)}")
        return False
def prepare_document(user_data: dict, processing_results: Dict) -> Dict:
    """Prepare document for Elasticsearch storage."""
    metadata = {
        "user_id": user_data['user_id'],
        "phone": user_data['phone'],
        "tweet_count": user_data['tweet_count']
    }
    
    # Since user_data['tweets'] is already a list of dicts, no need for .dict() conversion
    raw_tweets = user_data['tweets']
    
    return {
        "metadata": metadata,
        "processing_timestamp": datetime.utcnow().isoformat(),
        "tweet_analysis": processing_results,
        "raw_tweets": raw_tweets
    }

async def store_in_elasticsearch(document: Dict) -> bool:
    """Store document in Elasticsearch with retry logic."""
    max_retries = 3
    current_try = 0
    index_name = "tweets_analysis"

    while current_try < max_retries:
        try:
            doc_id = f"{document['metadata']['user_id']}_{int(datetime.utcnow().timestamp())}"
            
            # Convert any numpy types to Python native types
            document_json = json.loads(json.dumps(document, default=str))
            
            await es.index(
                index=index_name,
                id=doc_id,
                document=document_json,
                refresh=True
            )
            
            logger.info(f"Successfully stored document with ID: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Attempt {current_try + 1} failed: {str(e)}")
            if current_try == max_retries - 1:
                return False
            current_try += 1
    
    return False



@app.post("/process_user_tweets", response_model=Dict)
async def process_user_tweets_endpoint(request: Request):
    try:
        envelope = await request.json()
        message_data = base64.b64decode(envelope['message']['data']).decode('utf-8')
        user_data = json.loads(message_data)  


        tweets_df = pd.DataFrame([tweet for tweet in user_data['tweets']])
        tweets_df = tweets_df.rename(columns={'id': 'tweet_id'})
        
        tweets_df['views'] = pd.to_numeric(tweets_df['views'], errors='coerce')
        tweets_df['likes'] = pd.to_numeric(tweets_df['likes'], errors='coerce')
        tweets_df['shares'] = pd.to_numeric(tweets_df['shares'], errors='coerce')
        tweets_df['reply_count'] = pd.to_numeric(tweets_df['reply_count'], errors='coerce')

        processing_results = process_data(tweets_df)


        
        es_document = prepare_document(user_data, processing_results)

        storage_success = await store_in_elasticsearch(es_document)
        
        if not storage_success:
            logger.warning("Elasticsearch storage failed")
        
        return {
            "metadata": es_document["metadata"],
            "results": processing_results,
            "storage_status": "success" if storage_success else "failed"
        }
        
    except Exception as e:
        error_doc = {
            "metadata": {
                "user_id": user_data['user_id'] if 'user_data' in locals() and 'user_id' in user_data else 'unknown'
            },
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await store_in_elasticsearch({
                "metadata": error_doc["metadata"],
                "error_details": {
                    "error_message": str(e),
                    "timestamp": error_doc["timestamp"]
                }
            })
        except Exception as es_error:
            logger.error(f"Failed to store error in ES: {str(es_error)}")
        
        return JSONResponse(
            content=error_doc,
            status_code=400
        )
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)