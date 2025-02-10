
import json
import pandas as pd
import json
from transformers import pipeline

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
    # print("documents",documents)
    summarizer = pipeline(
        "summarization",
        model="google-t5/t5-small",  # Lightweight summarization model
    )

    # Generate summary
    summary = summarizer(documents, max_length=1000, min_length=50, do_sample=False)

    # print(summary[0])
    return summary[0]