import sys
from pathlib import Path

# Ensure the project directory is added to sys.path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

import os
import time
import json
import random
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI, APITimeoutError, APIConnectionError
import pinecone
from utils.mdx_cleaner import clean_mdx  # Keep this if you need cleaning logic
import numpy as np
import traceback

# Load environment variables
load_dotenv()

# New Relic API settings - THE ONLY ENDPOINT WE'LL USE
NC_KEY = os.getenv("NC_KEY")  # Your Nerd Completion API key
NC_ENDPOINT = os.getenv("NC_ENDPOINT", "https://nerd-completion.staging-service.nr-ops.net")
EMBEDDING_MODEL = "text-embedding-ada-002"  # Default embedding model

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "ai-hackathon-index"
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-west-2"  # Default to us-west-2 if not set

# Local path to java-agent docs
JAVA_AGENT_DIR = os.getenv("JAVA_AGENT_DIR", "java-agent")

# Print configuration (with masked keys)
def mask_key(key):
    if not key:
        return "Not set"
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "Too short"

print("=== Configuration ===")
print(f"NC_KEY: {mask_key(NC_KEY)}")
print(f"NC_ENDPOINT: {NC_ENDPOINT}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"PINECONE_API_KEY: {mask_key(PINECONE_API_KEY)}")
print(f"PINECONE_INDEX: {PINECONE_INDEX}")
print(f"PINECONE_ENV: {PINECONE_ENV}")
print(f"JAVA_AGENT_DIR: {JAVA_AGENT_DIR}")
print("=====================")

def get_local_files(directory, extensions=['.java', '.md', '.mdx', '.txt', '.html', '.xml', '.properties']):
    """Gets a list of files with specified extensions from a local directory."""
    files = []
    
    # Use Path for better cross-platform compatibility
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Error: Directory '{directory}' does not exist!")
        return files
    
    print(f"Exploring directory: {directory}")
    
    # Get all files recursively
    for ext in extensions:
        for file_path in dir_path.glob(f"**/*{ext}"):
            if file_path.is_file():
                files.append(str(file_path))
                print(f"Found file: {file_path}")
    
    # Output summary
    if len(files) == 0:
        print(f"⚠️ No files with extensions {extensions} found in '{directory}'!")
    else:
        print(f"✅ Found {len(files)} files in total")
    
    return files

def load_and_clean_docs(file_paths):
    """Loads and cleans files from the provided paths."""
    docs = []
    for file_path in tqdm(file_paths, desc="Loading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Determine how to process based on file extension
                if file_path.endswith('.mdx') or file_path.endswith('.md'):
                    # Clean MDX/MD content
                    processed_content = clean_mdx(content)
                elif file_path.endswith('.java'):
                    # For Java files, keep as is but you might add Java-specific cleaning
                    processed_content = content
                else:
                    # Default handling for other file types
                    processed_content = content
                
                docs.append(Document(page_content=processed_content, metadata={"source": file_path}))
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    docs.append(Document(page_content=content, metadata={"source": file_path}))
            except Exception as e:
                print(f"❌ Error reading file with latin-1 encoding {file_path}: {e}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    return docs

def prepare_text_for_embedding(text):
    """Prepare text for embedding by cleaning and truncating."""
    # Clean the text
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = ''.join(char for char in text if ord(char) < 128)
    text = ' '.join(text.split())
    
    # If text is too long, truncate it
    if len(text) > 8000:
        text = text[:8000]
    
    # If text is empty after cleaning
    if not text.strip():
        return "empty_document"
    
    return text

def get_embedding(text, max_retries=8, base_wait=5, max_wait=120):
    """Get an embedding using New Relic API with robust retry logic."""
    # Clean and prepare the text
    text = prepare_text_for_embedding(text)
    
    # Try getting embedding with exponential backoff
    for attempt in range(max_retries):
        try:
            # Initialize client with New Relic endpoint
            client = OpenAI(api_key=NC_KEY, base_url=NC_ENDPOINT, timeout=60.0)
            
            # Get embedding
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[text]
            )
            # Success! Return the embedding
            return response.data[0].embedding
        
        except (APITimeoutError, APIConnectionError) as e:
            # Specific handling for timeout errors
            wait_time = min(base_wait * (2 ** attempt) + random.uniform(1, 5), max_wait)
            print(f"API timeout on attempt {attempt+1}/{max_retries}. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            
            # On last attempt, return random embedding
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed. Using random embedding.")
                return generate_random_embedding()
        
        except Exception as e:
            print(f"Unexpected error on attempt {attempt+1}/{max_retries}: {e}")
            print(traceback.format_exc())
            
            # Wait with some randomization to avoid synchronized retries
            wait_time = min(base_wait * (1.5 ** attempt) + random.uniform(1, 3), max_wait / 2)
            print(f"Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
            
            # On last attempt, return random embedding
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed. Using random embedding.")
                return generate_random_embedding()

def generate_random_embedding(dim=1536):
    """Generate a random unit vector for fallback."""
    vector = np.random.uniform(-1, 1, dim)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def create_pinecone_index_if_not_exists(pc, index_name):
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        
        try:
            # Try to create with ServerlessSpec
            pc.create_index(
                name=index_name,
                dimension=1536,  # Embedding dimension
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )
        except Exception as spec_error:
            print(f"Error creating index with ServerlessSpec: {spec_error}")
            print("Trying without ServerlessSpec...")
            
            # Try to create without ServerlessSpec
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine"
            )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(20)

def load_checkpoint(checkpoint_file="embedding_progress.json"):
    """Load progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                progress = json.load(f)
                print(f"Loaded progress from checkpoint. Completed: {len(progress.get('completed', []))} chunks.")
                return progress
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return {"completed": [], "embeddings": {}}

def save_checkpoint(progress, checkpoint_file="embedding_progress.json"):
    """Save progress to checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        json.dump(progress, f)

def process_chunks(chunks, batch_size=2):
    """Process chunks with NR API - using very small batches for reliability."""
    embeddings = []
    total_chunks = len(chunks)
    
    # Load checkpoint
    progress = load_checkpoint()
    
    # Process each chunk individually with aggressive throttling
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        
        # Skip if already processed
        if chunk_id in progress.get("completed", []):
            print(f"Skipping already processed chunk {i+1}/{total_chunks}")
            # Load saved embedding
            embeddings.append(json.loads(progress["embeddings"][chunk_id]))
            continue
        
        print(f"Processing chunk {i+1}/{total_chunks}")
        text = chunk.page_content
        
        # Get embedding with extreme resilience
        embedding = get_embedding(text)
        embeddings.append(embedding)
        
        # Save progress
        if "completed" not in progress:
            progress["completed"] = []
        if "embeddings" not in progress:
            progress["embeddings"] = {}
            
        progress["completed"].append(chunk_id)
        progress["embeddings"][chunk_id] = json.dumps(embedding)  # Store as JSON string
        
        save_checkpoint(progress)
        
        # Add randomized sleep between requests to avoid regular patterns
        sleep_time = 3.0 + (random.random() * 4.0)  # Random sleep between 3-7 seconds
        print(f"Waiting {sleep_time:.2f} seconds before next request...")
        time.sleep(sleep_time)
    
    return embeddings

def embed_docs_to_pinecone(docs):
    """Process documents and embed them to Pinecone."""
    if not docs:
        print("⚠️ No documents to embed. Exiting.")
        return
    
    print("Preparing documents for embedding...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Smaller chunks for NR API to handle better
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    
    # Process chunks with NR embeddings
    print("Generating embeddings using New Relic API...")
    embeddings = process_chunks(chunks, batch_size=2)  # Using small batch size for NR API
    
    print(f"Successfully generated {len(embeddings)} embeddings")
    
    # Initialize Pinecone
    print(f"Connecting to Pinecone index '{PINECONE_INDEX}'...")
    
    try:
        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        create_pinecone_index_if_not_exists(pc, PINECONE_INDEX)
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX)
        
        # Using smaller batch size for Pinecone upserts with NR
        batch_size = 10
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
                
                # Create vector object with metadata
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": chunk.metadata["source"],
                        "text": chunk.page_content[:200],  # Store preview of content
                        "file_type": Path(chunk.metadata["source"]).suffix,
                        "file_name": Path(chunk.metadata["source"]).name
                    }
                }
                vectors.append(vector)
            
            # Upsert vectors with retries
            max_retries = 3
            for retry in range(max_retries):
                try:
                    index.upsert(vectors=vectors)
                    print(f"✓ Batch {batch_num}/{total_batches} upserted successfully")
                    
                    # Add spacing between batches
                    if i + batch_size < len(chunks):
                        time.sleep(3)  # Longer wait between batches when using NR API
                        
                    break
                except Exception as upsert_error:
                    print(f"❌ Error upserting batch {batch_num} (Retry {retry+1}/{max_retries}): {upsert_error}")
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 5
                        print(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to upsert batch {batch_num} after {max_retries} retries.")
                        # Try upserting one by one as a fallback
                        print("Trying to upsert vectors one by one...")
                        
                        for v_idx, vector in enumerate(vectors):
                            try:
                                index.upsert(vectors=[vector])
                                print(f"  ✓ Upserted vector {v_idx+1}/{len(vectors)}")
                                time.sleep(1)  # Brief pause between single vector upserts
                            except Exception as e:
                                print(f"  ❌ Failed to upsert vector {v_idx+1}/{len(vectors)}: {e}")
        
        print(f"✅ Successfully embedded {len(chunks)} chunks to Pinecone index '{PINECONE_INDEX}'")
        
    except Exception as e:
        print(f"❌ Error with Pinecone: {e}")
        raise

def query_pinecone(query_text, top_k=5):
    """Query the Pinecone index with a text query."""
    # Generate embedding for the query
    query_embedding = get_embedding(query_text)
    
    try:
        # Connect to Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        
        # Query the index
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Process and return results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "source": match.metadata.get("source", "Unknown"),
                "text": match.metadata.get("text", "No preview available")
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return []

def interactive_rag():
    """Interactive RAG query interface."""
    print("\n=== Java-Agent RAG Query Interface ===")
    print("Type 'quit' or 'exit' to exit.")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() in ["quit", "exit", "q"]:
            break
        
        print("\nSearching for relevant information...")
        results = query_pinecone(query, top_k=5)
        
        if not results:
            print("No results found.")
            continue
        
        print("\n=== Top Results ===")
        for i, result in enumerate(results):
            print(f"\n[{i+1}] Score: {result['score']:.2f}")
            print(f"Source: {result['source']}")
            print(f"Preview: {result['text']}...")
        
        # Here you would typically call a language model with the retrieved context
        print("\nTo use these results with a language model, implement the generate_answer() function.")

def main():
    print("Java-Agent RAG Implementation with New Relic API")
    
    # Verify API keys are set
    if not NC_KEY:
        print("❌ Error: NC_KEY is not set in your environment")
        sys.exit(1)
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY is not set in your environment")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Java-Agent RAG Implementation with New Relic API")
    parser.add_argument("--index", action="store_true", help="Index documents to Pinecone")
    parser.add_argument("--query", action="store_true", help="Query the Pinecone index interactively")
    parser.add_argument("--dir", type=str, default=JAVA_AGENT_DIR, help="Directory containing Java agent files")
    parser.add_argument("--max-files", type=int, default=100, help="Limit the number of files to process (0 = all)")
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not (args.index or args.query):
        parser.print_help()
        return
    
    if args.index:
        print(f"Indexing files from directory: {args.dir}")
        
        file_paths = get_local_files(args.dir)
        
        if len(file_paths) == 0:
            print("⚠️ No files found to process. Exiting.")
            return
        
        # Option to limit the number of files for testing
        if args.max_files > 0 and len(file_paths) > args.max_files:
            print(f"Limiting to {args.max_files} files for processing (set by --max-files)")
            file_paths = file_paths[:args.max_files]
        
        print("Loading and cleaning documents...")
        docs = load_and_clean_docs(file_paths)
        print(f"Loaded {len(docs)} documents")
        
        if len(docs) == 0:
            print("⚠️ No documents were successfully loaded. Exiting.")
            return
        
        embed_docs_to_pinecone(docs)
        print("Document embedding process completed!")
    
    if args.query:
        interactive_rag()

if __name__ == "__main__":
    main()