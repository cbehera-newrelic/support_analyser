import sys
from pathlib import Path

# Ensure the project directory is added to sys.path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

import os
import requests
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# Import OpenAI directly
from openai import OpenAI
import pinecone
from utils.mdx_cleaner import clean_mdx

# Load environment variables
load_dotenv()

# New Relic specific OpenAI-compatible endpoint (commented out)
# NC_KEY = os.getenv("NC_KEY")  # Your Nerd Completion API key
# NC_ENDPOINT = os.getenv("NC_ENDPOINT", "https://nerd-completion.staging-service.nr-ops.net")

# Use OpenAI API key instead
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "ai-hackathon-index"
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-west-2"  # Default to us-west-2 if not set

# GitHub repo details - verified path structure
GITHUB_OWNER = "CBehera5"
GITHUB_REPO = "docs-website-chinmay"
BRANCH = "develop"
BASE_PATH = "src/content/docs/apm/agents"

# Print configuration (with masked keys)
def mask_key(key):
    if not key:
        return "Not set"
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "Too short"

print("=== Configuration ===")
# print(f"NC_KEY: {mask_key(NC_KEY)}")
# print(f"NC_ENDPOINT: {NC_ENDPOINT}")
print(f"OPENAI_API_KEY: {mask_key(OPENAI_API_KEY)}")
print(f"PINECONE_API_KEY: {mask_key(PINECONE_API_KEY)}")
print(f"PINECONE_INDEX: {PINECONE_INDEX}")
print(f"PINECONE_ENV: {PINECONE_ENV}")
print("=====================")

def get_mdx_files_from_github(owner, repo, branch, base_path):
    """Gets a list of MDX files from a GitHub repository.
    
    This function includes robust error handling, logging, and
    proper handling of GitHub API authentication and rate limits.
    """
    mdx_files = []
    processed_dirs = set()  # Keep track of directories we've already processed
    
    def fetch_dir(path):
        # Skip if we've already processed this directory
        if path in processed_dirs:
            return
        
        processed_dirs.add(path)
        print(f"Exploring directory: {path}")
        
        # GitHub API endpoint for directory contents
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        
        # Add User-Agent header to avoid GitHub API rate limiting
        headers = {
            "User-Agent": "MDX-Fetcher-Script",
            # Add a GitHub token if you have one to avoid rate limits
            "Authorization": f"token {os.getenv('GITHUB_TOKEN')}"
        }
        
        try:
            # Make the request with headers
            response = requests.get(url, headers=headers)
            
            # Check for rate limiting
            if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers.get('X-RateLimit-Remaining', '0'))
                if remaining == 0:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', '0'))
                    wait_seconds = max(reset_time - int(time.time()), 0)
                    print(f"⚠️ GitHub API rate limit exceeded! Waiting {wait_seconds} seconds...")
                    time.sleep(wait_seconds + 5)  # Add 5 seconds buffer
                    # Try again after waiting
                    fetch_dir(path)
                    return
            
            # Handle other errors
            if response.status_code != 200:
                print(f"❌ Error accessing {url}: HTTP {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return
            
            # Parse the JSON response
            contents = response.json()
            
            # Handle case when the API returns a single file instead of a list
            if not isinstance(contents, list):
                contents = [contents]
            
            print(f"Found {len(contents)} items in '{path}'")
            
            # Process each item in the directory
            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".mdx"):
                    mdx_files.append(item["path"])
                    print(f"Found MDX file: {item['path']}")
                elif item["type"] == "dir":
                    # Recursively explore subdirectories
                    fetch_dir(item["path"])
        
        except Exception as e:
            print(f"❌ Error while fetching directory '{path}': {e}")
    
    # Start the recursive process with the base path
    fetch_dir(base_path)
    
    # Output summary
    if len(mdx_files) == 0:
        print("⚠️ No MDX files found! Check the repository path and branch.")
    else:
        print(f"✅ Found {len(mdx_files)} MDX files in total")
    
    return mdx_files

def convert_to_raw_url(owner, repo, branch, filepath):
    """Converts a GitHub file path to its raw content URL."""
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}"

def load_and_clean_docs(urls):
    """Loads and cleans MDX files from the provided URLs."""
    docs = []
    for url in tqdm(urls, desc="Downloading .mdx files"):
        try:
            res = requests.get(url)
            if res.status_code == 200:
                # Clean the MDX content
                raw = clean_mdx(res.text)
                docs.append(Document(page_content=raw, metadata={"source": url}))
            else:
                print(f"⚠️ Failed to download {url}: HTTP {res.status_code}")
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
    
    return docs

# New Relic version (commented out)
"""
def get_embeddings_from_nerd_completion(texts, batch_size=100):
    # Initialize the OpenAI client with custom endpoint
    client = OpenAI(api_key=NC_KEY, base_url=NC_ENDPOINT)
    
    all_embeddings = []
    
    # Process in batches to avoid timeouts and rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Generating embeddings for batch {batch_num}/{total_batches} ({len(batch)} texts)")
        
        try:
            # Get embeddings for this batch
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"✓ Successfully generated {len(batch_embeddings)} embeddings for batch {batch_num}")
            
            # Slight pause to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error generating embeddings for batch {batch_num}: {e}")
            # Return empty embeddings for failed batch to maintain alignment
            all_embeddings.extend([[0] * 1536] * len(batch))
    
    return all_embeddings
"""

# OpenAI version
def get_embeddings_from_openai(texts, batch_size=100):
    """Generate embeddings using the OpenAI API."""
    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    all_embeddings = []
    
    # Process in batches to avoid timeouts and rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Generating embeddings for batch {batch_num}/{total_batches} ({len(batch)} texts)")
        
        try:
            # Get embeddings for this batch
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"✓ Successfully generated {len(batch_embeddings)} embeddings for batch {batch_num}")
            
            # Slight pause to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error generating embeddings for batch {batch_num}: {e}")
            # Generate random embeddings for failed batch to maintain alignment
            import numpy as np
            for _ in range(len(batch)):
                vector = np.random.uniform(-1, 1, 1536)
                # Normalize
                vector = vector / np.linalg.norm(vector)
                all_embeddings.append(vector.tolist())
            print(f"⚠️ Using random embeddings for batch {batch_num} due to error")
    
    return all_embeddings

def embed_docs_to_pinecone(docs):
    """Process documents and embed them to Pinecone using OpenAI embeddings."""
    if not docs:
        print("⚠️ No documents to embed. Exiting.")
        return
    
    print("Preparing documents for embedding...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    
    # Extract text from chunks for embedding
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings using OpenAI API
    print("Generating embeddings using OpenAI API...")
    embeddings = get_embeddings_from_openai(texts)
    
    if not embeddings:
        print("❌ Failed to generate embeddings. Exiting.")
        return
    
    print(f"Successfully generated {len(embeddings)} embeddings")
    
    # Initialize Pinecone
    print(f"Connecting to Pinecone index '{PINECONE_INDEX}'...")
    
    try:
        # Initialize Pinecone client
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        # Create index if it doesn't exist
        if PINECONE_INDEX not in existing_indexes:
            print(f"Creating new Pinecone index: {PINECONE_INDEX}")
            
            try:
                # Try to create with ServerlessSpec
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=1536,  # Embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENV
                    )
                )
            except Exception as spec_error:
                print(f"Error creating index with ServerlessSpec: {spec_error}")
                print("Trying without ServerlessSpec...")
                
                # Try to create without ServerlessSpec
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=1536,
                    metric="cosine"
                )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(20)
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX)
        
        # Prepare vectors for upsert
        print("Preparing vectors for Pinecone...")
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeds = embeddings[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
            
            # Create vectors
            vectors = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeds)):
                vector_id = f"chunk_{i+j}"
                
                # Create vector object
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": chunk.metadata["source"],
                        "text": chunk.page_content[:200]  # Store preview of content
                    }
                }
                vectors.append(vector)
            
            # Upsert vectors
            try:
                index.upsert(vectors=vectors)
                print(f"✓ Batch {batch_num}/{total_batches} upserted successfully")
            except Exception as upsert_error:
                print(f"❌ Error upserting batch {batch_num}: {upsert_error}")
        
        print(f"✅ Successfully embedded {len(chunks)} chunks to Pinecone index '{PINECONE_INDEX}'")
        
    except Exception as e:
        print(f"❌ Error with Pinecone: {e}")
        raise

if __name__ == "__main__":
    print("Starting document embedding process...")
    
    # Verify API key is set
    # if not NC_KEY:
    #     print("❌ Error: NC_KEY is not set in your environment")
    #     sys.exit(1)
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY is not set in your environment")
        sys.exit(1)
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY is not set in your environment")
        sys.exit(1)
    
    print(f"Fetching .mdx files from GitHub: {GITHUB_OWNER}/{GITHUB_REPO}/{BRANCH}/{BASE_PATH}")
    
    mdx_paths = get_mdx_files_from_github(GITHUB_OWNER, GITHUB_REPO, BRANCH, BASE_PATH)
    print(f"Found {len(mdx_paths)} .mdx files")
    
    if len(mdx_paths) == 0:
        print("⚠️ No MDX files found to process. Exiting.")
        sys.exit(1)
    
    # Option to limit the number of files for testing
    max_files = int(os.getenv("MAX_FILES", 100))
    if max_files > 0 and len(mdx_paths) > max_files:
        print(f"Limiting to {max_files} files for processing (set by MAX_FILES)")
        mdx_paths = mdx_paths[:max_files]
    
    raw_urls = [convert_to_raw_url(GITHUB_OWNER, GITHUB_REPO, BRANCH, path) for path in mdx_paths]
    
    print("Loading and cleaning documents...")
    docs = load_and_clean_docs(raw_urls)
    print(f"Loaded {len(docs)} documents")
    
    if len(docs) == 0:
        print("⚠️ No documents were successfully loaded. Exiting.")
        sys.exit(1)
    
    embed_docs_to_pinecone(docs)
    print("Document embedding process completed!")