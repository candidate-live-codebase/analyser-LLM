import json
import pandas as pd
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain_core.documents import Document 
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from typing import List, Dict

# ---------------------- Metadata Extraction Function ---------------------- #
def metadata_func(record, metadata):
    metadata["id"] = record.get("id")
    metadata["datetime"] = record.get("datetime")
    metadata["user_name"] = record.get("username")  # Adjusted field name to 'username'
    metadata["views"] = record.get("views")
    metadata["likes"] = record.get("likes")
    metadata["shares"] = record.get("shares")
    metadata["reply_count"] = record.get("reply_count")
    return metadata

# ---------------------- Convert DataFrame to Documents ---------------------- #
def convert_df_to_documents(df: pd.DataFrame) -> List[Document]:
    """Convert the DataFrame of tweets into a list of LangChain Document objects."""
    documents = []
    for _, row in df.iterrows():
        metadata = metadata_func(row.to_dict(), {})  # Extract metadata
        content = row.get("content", "")

        # Create a LangChain Document object
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    return documents

# ---------------------- Process Documents ---------------------- #
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# ---------------------- Load Ollama Model ---------------------- #
def load_ollama_model():
    return Ollama(model="gemma2:9b-instruct-q4_0")

# ---------------------- Summarization Prompt ---------------------- #
def create_summary_prompt():
    return PromptTemplate(
        input_variables=["text", "metadata"],
        template="Generate key insights and topics in a concise and clear manner and in 4 pointers and give key points for positive and negative content, based on engagement: Note: summary should be in 200 words\n\n{text}",
    )

# ---------------------- Summarization Chain ---------------------- #
def generate_summary(documents):
    llm = load_ollama_model()
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(documents)

# Main function to process and summarize data
def summarize_json(tweets_df):
    # # Convert the incoming user_data['tweets'] to DataFrame
    # tweets_df = pd.DataFrame(user_data['tweets'])
    
    # Convert the DataFrame to document-like structure
    documents = convert_df_to_documents(tweets_df)
    
    # Process the documents into chunks
    split_docs = process_documents(documents)
    
    # Generate the summary
    summary = generate_summary(split_docs)
    
    return summary
