import json
import pandas as pd
import json
from transformers import pipeline
import anthropic

from datetime import datetime


from langchain_community.document_loaders import JSONLoader


# ---------------------- Convert DataFrame to Documents ---------------------- #
def convert_df_to_documents(df: pd.DataFrame) -> str:
    """Convert the DataFrame tweets into a consolidated text string."""
    all_text = []
    
    for _, row in df.iterrows():
        content = row.get("content", "").strip()
        if content:  # Only add non-empty content
            all_text.append(content)
    
    return " ".join(all_text)



def summarize_json(tweets_df):
  
    documents = convert_df_to_documents(tweets_df)

    prompt=f"""Human: You are a summarisation assistant. Your task is to summarise product reviews given to you as a list. Within this list, there are individual product reviews in an array.
          Create a JSON document with the following fields:
          summary - A summary of these reviews in less than 250 words
          overall_sentiment - The overall sentiment of the reviews
          sentiment_confidence - How confident you are about the sentiment of the reviews
          reviews_positive - The percent of positive reviews
          reviews_neutral - The percent of neutral reviews
          reviews_negative - The percent of negative reviews
          action_items - A list of action items to resolve the customer complaints (don't put soemthing which is already good and there is no customer complaints)
          Your output should be raw JSON - do not include any sentences or additional text outside of the JSON object.
          Here is the list of reviews that I want you to summarise:
          {documents}
          Assistant:"""

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="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",
    )
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    output = message.content

    analysis=json.loads(output[0].text)
    return  analysis
