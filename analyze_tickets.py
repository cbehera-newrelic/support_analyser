import os
import pandas as pd
import streamlit as st
import json
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
import random
import time
import traceback

# Set Streamlit page config first, before any other Streamlit commands
st.set_page_config(page_title="Support Ticket Analyzer", layout="wide")

# Load environment variables
load_dotenv()

# New Relic API settings - ONLY USING NEW RELIC ENDPOINT
NC_KEY = os.getenv("NC_KEY")  # Your Nerd Completion API key
NC_ENDPOINT = os.getenv("NC_ENDPOINT", "https://nerd-completion.staging-service.nr-ops.net")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "hackathon"

# Check if New Relic API key is available
if not NC_KEY:
    st.error("No New Relic API key found. Please set NC_KEY in your .env file.")
    st.stop()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Function to generate embeddings using New Relic's endpoint with resilient retries
def get_embedding(text, max_retries=5, base_wait=2, max_wait=60):
    for attempt in range(max_retries):
        try:
            # Check for NaN or empty text
            if pd.isna(text) or text == "":
                st.warning(f"Empty or NaN text encountered. Using random embedding.")
                return generate_random_embedding()
            
            # Clean the text to remove any problematic characters
            # Replace line breaks, tabs, etc.
            text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Remove any non-ASCII characters
            text = ''.join(char for char in text if ord(char) < 128)
            # Remove extra spaces
            text = ' '.join(text.split())
            
            # If text is still empty after cleaning
            if not text.strip():
                st.warning(f"Text was empty after cleaning. Using random embedding.")
                return generate_random_embedding()
            
            # Truncate very long texts to prevent timeout issues
            if len(text) > 8000:
                text = text[:8000]
                
            # Initialize client with New Relic endpoint
            client = OpenAI(api_key=NC_KEY, base_url=NC_ENDPOINT, timeout=60.0)
            
            # Get embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=[text]
            )
            return response.data[0].embedding
            
        except Exception as e:
            wait_time = min(base_wait * (2 ** attempt), max_wait)
            
            if attempt < max_retries - 1:
                st.warning(f"New Relic embedding API error on attempt {attempt+1}/{max_retries}: {e}. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                st.warning(f"All {max_retries} embedding attempts failed: {e}. Falling back to random embeddings.")
                return generate_random_embedding()

# Function to generate a random embedding for fallback
def generate_random_embedding(dim=1536):
    vector = [random.uniform(-1, 1) for _ in range(dim)]
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [float(x/magnitude) for x in vector]

# Simple retriever function
def get_relevant_documents(query_text, top_k=3):
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(query_text)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Convert to document-like objects
        documents = []
        for match in results.matches:
            metadata = match.metadata
            doc = type('Document', (), {
                'page_content': metadata.get('text', ''),
                'metadata': metadata
            })
            documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        st.error(traceback.format_exc())
        return []

# Function to get chat completion from New Relic with resilient retries
def get_chat_completion(prompt, max_retries=3, base_wait=3, max_wait=30):
    for attempt in range(max_retries):
        try:
            # Initialize client with New Relic endpoint
            client = OpenAI(api_key=NC_KEY, base_url=NC_ENDPOINT, timeout=90.0)
            
            # Get completion
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            wait_time = min(base_wait * (2 ** attempt), max_wait)
            
            if attempt < max_retries - 1:
                st.warning(f"New Relic chat API error on attempt {attempt+1}/{max_retries}: {e}. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                st.error(f"All {max_retries} chat completion attempts failed: {e}")
                return f"ERROR: Could not generate analysis after {max_retries} attempts. API error: {e}"

# Title and header
st.title("🧠 Support Ticket Analyzer: Doc or UI Fix?")

# Check system status in sidebar
with st.sidebar:
    st.subheader("System Status")
    
    # Check Pinecone
    try:
        stats = index.describe_index_stats()
        st.success(f"✅ Pinecone connected: {stats.total_vector_count} vectors in index")
    except Exception as e:
        st.error(f"❌ Pinecone error: {e}")
    
    # Check New Relic API
    with st.spinner("Checking New Relic API..."):
        try:
            # Try a simple embedding request
            test_result = get_embedding("test")
            if isinstance(test_result, list) and len(test_result) > 0:
                st.success("✅ New Relic API working")
            else:
                st.warning("⚠️ New Relic API returned unexpected result")
        except Exception as e:
            st.error(f"❌ New Relic API error: {e}")

# Main UI
ticket_file = st.file_uploader("📄 Upload Support Tickets CSV", type=["csv"])

# Add encoding options for CSV reading
encoding_options = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
selected_encoding = st.selectbox("CSV Encoding", encoding_options, index=0)

if ticket_file:
    try:
        # Show preview of the uploaded file
        df_preview = pd.read_csv(ticket_file, encoding=selected_encoding)
        st.subheader("Preview of uploaded tickets")
        st.dataframe(df_preview.head(3), use_container_width=True)
        
        # Show column names to help with mapping
        st.subheader("Available Columns")
        st.write(df_preview.columns.tolist())
        
        # Count of tickets
        st.info(f"📊 Total tickets: {len(df_preview)}")
    except Exception as e:
        st.error(f"Error reading CSV with {selected_encoding} encoding: {e}")
        st.info("Please try a different encoding from the dropdown above.")

if st.button("🔍 Analyze Tickets") and ticket_file:
    # Reset file pointer
    ticket_file.seek(0)
    
    try:
        # Load and prepare data
        df = pd.read_csv(ticket_file, encoding=selected_encoding)
        
        # Column mapping (adjust to match your actual column names)
        title_column = "Title"  # Your actual title column name
        description_column = "Description"  # Your actual description column name
        ticket_no_column = "Ticket No."  # Your ticket number column
        
        # Column mapping UI
        st.subheader("Column Mapping")
        cols = st.columns(3)
        with cols[0]:
            title_column = st.selectbox("Title Column", options=df.columns, index=df.columns.get_loc(title_column) if title_column in df.columns else 0)
        with cols[1]:
            description_column = st.selectbox("Description Column", options=df.columns, index=df.columns.get_loc(description_column) if description_column in df.columns else 0)
        with cols[2]:
            ticket_no_column = st.selectbox("Ticket ID Column", options=df.columns, index=df.columns.get_loc(ticket_no_column) if ticket_no_column in df.columns else 0)
        
        # Clean the DataFrame before processing
        for col in [title_column, description_column]:
            if col in df.columns:
                # Replace NaN values with empty strings
                df[col] = df[col].fillna("")
                # Convert all values to strings
                df[col] = df[col].astype(str)
                # Remove any non-ASCII characters
                df[col] = df[col].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
                # Replace multiple spaces with a single space
                df[col] = df[col].apply(lambda x: ' '.join(x.split()))
        
        # Create combined text with better error handling
        df['combined'] = df[title_column].fillna("").astype(str) + ". " + df[description_column].fillna("").astype(str)
        
        # Additional step: Check for and clean any problematic data
        df['combined'] = df['combined'].apply(lambda x: ' '.join(str(x).split()) if not pd.isna(x) else "")
        ticket_texts = df['combined'].tolist()
        ticket_ids = df[ticket_no_column].tolist() if ticket_no_column in df.columns else [f"Ticket-{i+1}" for i in range(len(df))]

        # Process all tickets by default
        max_tickets = st.number_input("Maximum tickets to process", min_value=1, max_value=len(ticket_texts), value=min(10, len(ticket_texts)))
        ticket_texts = ticket_texts[:max_tickets]
        ticket_ids = ticket_ids[:max_tickets]
        
        results = []
        
        # Add progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for idx, (ticket_id, ticket) in enumerate(zip(ticket_ids, ticket_texts)):
            try:
                # Update progress
                progress_text.text(f"Processing ticket {idx+1}/{len(ticket_texts)}")
                progress_bar.progress((idx + 1) / len(ticket_texts))
                
                # Get relevant documents
                docs = get_relevant_documents(ticket)
                
                # Create context from documents
                context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documentation found."
                
                # Create the prompt with JSON structure for better parsing
                prompt = f"""
You are the Support Ticket Analyzer Agent. Analyze the provided support ticket and documentation context using a step-by-step Chain of Thought (CoT). Your goal is to categorize the ticket and identify its root cause.

Inputs:

Ticket: {ticket}
Context Docs: {context}
CoT Process - Think step-by-step:

1. Understand Ticket & Context:
* Grasp the core customer issue from the {ticket}.
* Review the {context} docs and note their relevance to the ticket.

2. Hierarchical Categorization (based on {ticket}):
* a. Product Line: Identify the main product.
* b. Capability: Determine the broad product capability involved.
* c. Feature: Pinpoint the specific feature.
* d. Action/Problem: Describe the customer's action or specific problem (e.g., Installation, Pricing inquiry, Cannot find UI element, Error during X).

3. Root Cause Analysis (using {ticket} and {context}):
* Determine the Primary_Cause_Type. Justify your choice based on evidence from both {ticket} and your assessment of {context}.
* Documentation Issue? (Docs missing, unclear, incorrect? Does {context} address the issue well? If not, is that the gap?)
* UI/UX Issue? (Confusing interface, element not working, poor usability, even if {context} seems relevant?)
* Pricing Issue? (Ticket primarily about cost, subscription, billing?)
* Other Product Issue? (Core bug, backend error, performance, feature malfunction not covered above?)

4. Confidence (Internal Thought):
* Assess your confidence ("High", "Medium", "Low") for categorization and root cause.

5. JSON Output:
* Format your response as JSON with the following structure: 
{{
  "category": "The hierarchical category from step 2",
  "primary_issue": "Documentation Issue|UI/UX Issue|Pricing Issue|Other Product Issue",
  "explanation": "Explain the issue and why you categorized it this way",
  "recommendation": "What you recommend to fix this issue (doc update, UI fix, etc.)"
}}
"""
                # Get completion
                reply = get_chat_completion(prompt)
                
                try:
                    # Try to parse the response as JSON
                    parsed_response = json.loads(reply)
                    
                    # Create a more structured result
                    results.append({
                        "Ticket ID": ticket_id,
                        "Ticket Text": ticket[:200] + "..." if len(ticket) > 200 else ticket,
                        "Category": parsed_response.get("category", "Unknown"),
                        "Primary Issue": parsed_response.get("primary_issue", ""),
                        "Explanation": parsed_response.get("explanation", ""),
                        "Recommendation": parsed_response.get("recommendation", ""),
                        "Raw Response": reply  # Keep the full response for reference
                    })
                except json.JSONDecodeError:
                    # If the response isn't valid JSON, keep the original format
                    results.append({
                        "Ticket ID": ticket_id,
                        "Ticket Text": ticket[:200] + "..." if len(ticket) > 200 else ticket,
                        "Recommendation": reply
                    })
                    st.warning(f"Response for ticket {ticket_id} wasn't in the expected format.")
                
                # Add a small delay to avoid rate limits - increased for New Relic API
                time.sleep(2.0)
                
            except Exception as e:
                results.append({
                    "Ticket ID": ticket_id,
                    "Ticket Text": ticket[:200] + "..." if len(ticket) > 200 else ticket,
                    "Recommendation": f"Error: {e}"
                })
                st.warning(f"Error processing ticket {idx+1}: {e}")
        
        # Display the results in a more structured way
        st.success("✅ Analysis complete.")

        # First show a summary table with just the key information
        summary_df = pd.DataFrame({
            "Ticket ID": [r.get("Ticket ID") for r in results],
            "Category": [r.get("Category", "N/A") for r in results],
            "Primary Issue": [r.get("Primary Issue", "N/A") for r in results]
        })
        st.subheader("Summary of Analysis")
        st.dataframe(summary_df, use_container_width=True)

        # Then show detailed results with expandable sections
        st.subheader("Detailed Analysis")
        for i, result in enumerate(results):
            issue_title = result.get('Primary Issue', 'Analysis')
            if not issue_title or issue_title == "N/A":
                issue_title = "Analysis"
                
            with st.expander(f"Ticket {result.get('Ticket ID')} - {issue_title}"):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("#### Ticket")
                    st.markdown(f"**ID:** {result.get('Ticket ID')}")
                    st.markdown(f"**Text:** {result.get('Ticket Text', '')}")
                
                with cols[1]:
                    st.markdown("#### Analysis")
                    st.markdown(f"**Category:** {result.get('Category', 'N/A')}")
                    st.markdown(f"**Primary Issue:** {result.get('Primary Issue', 'N/A')}")
                
                st.markdown("#### Explanation")
                st.markdown(result.get("Explanation", "N/A"))
                
                st.markdown("#### Recommendation")
                st.markdown(result.get("Recommendation", "N/A"))
                
                if "Raw Response" in result:
                    with st.expander("View Raw Response"):
                        st.text(result.get("Raw Response", ""))

        # Add download options
        st.subheader("Download Results")
        csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Analysis CSV", csv, "ticket_analysis.csv", "text/csv")

        # Add option to download just the summary
        summary_csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary CSV", summary_csv, "ticket_summary.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Error processing tickets: {e}")
        st.error(traceback.format_exc())
        st.write("Please check your CSV file format and try again.")
else:
    st.info("👆 Please upload the support CSV file and click 'Analyze Tickets'.")