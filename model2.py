import json
import openai
import os
import json
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime


from langchain_community.document_loaders import JSONLoader
os.environ["OPENAI_API_KEY"] = "your api key"

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("API key not found")

# ---------------------- Convert DataFrame to Documents ---------------------- #
def convert_df_to_documents(df: pd.DataFrame) -> str:
    """Convert the DataFrame reviews into a consolidated text string."""
    all_text = []
    
    for _, row in df.iterrows():
        content = row.get("content", "").strip()
        if content:  # Only add non-empty content
            all_text.append(content)
    
    return " ".join(all_text)

def summarize_json(tweets_df: pd.DataFrame):
    """Generate a structured JSON summary using GPT-4o."""
    documents = convert_df_to_documents(tweets_df)

    if not documents:
        raise ValueError("No content found in the 'content' column")

    prompt = f"""Human: You are a summarisation assistant. Your task is to summarise product reviews given to you as a list. Within this list, there are individual product reviews in an array.
          Create a JSON document with the following fields:
          summary - A summary of these reviews in less than 250 words in pointers
          overall_sentiment - The overall sentiment of the reviews
          sentiment_confidence - How confident you are about the sentiment of the reviews
          reviews_positive - The percent of positive reviews
          reviews_neutral - The percent of neutral reviews
          reviews_negative - The percent of negative reviews
          action_items - A list of action items to resolve the customer complaints (don't put something which is already good and there is no customer complaint ), you can put any suggestive item which can help to improve sentiment
          Your output should be raw JSON - do not include any sentences or additional text outside of the JSON object.
          Here is the list of reviews that I want you to summarise:
          {documents}
          Assistant:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        content = response["choices"][0]["message"].get("content", "").strip()

        if not content:
            raise ValueError("Empty response from the model")
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]  # Remove ending ```
        analysis = json.loads(content)
        return analysis

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response content: {content}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
