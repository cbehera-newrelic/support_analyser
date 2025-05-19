# support_analyser
The Support Analysis Tool is an AI-driven solution designed to automate the analysis of support tickets, enhancing the efficiency and responsiveness of Global Technical Support. Utilizing agentic AI and RAG (Retrieval-Augmented Generation) methodologies, it processes support tickets in batches, analyzes them using a knowledge base (specifically a document repository), and generates the following insights for each ticket:

* **Hierarchical Categorization**: Relevant product areas and sub-categories of the issue.
* **Root cause**: underlying cause of the issue if the following format:
    * Primary_Cause_Type
    * Cause_Justification
    * Documentation issue/UI/UX issue/pricing issue/other
    * Details, if it is an other product issue.

## Pre-requisites
* A .csv file with support tickets
* A folder with the knowledge articles (Utilized our public documentation articles for this purpose)
* Your NR API key
* Your github PAT
* Your pine cone API key for vector embedding storage

## Installation Instructions

1. Clone the [repository](https://github.com/cbehera-newrelic/support_analyser) to your local machine.
2. On your local machine, open the cloned folder in a code editor.
3. Upload the knowledge article folder in the `support_analyser` repo.
4. In the terminal, perform the following steps:

   
    a. Create a virtual environment:
   
       ```
       python3 -m venv venv
       source venv/bin/activate
    
       ```
    b. Install dependencies:
   
       ```
       pip install -r requirements.txt
   
       ```
   c. Inside your repo, create a .env file, and enter the following:

        ```
        NC_KEY=<Your NR API key>
        TEAM_NAME=<Your team name>
        GITHUB_TOKEN=<Your github PAT>
        PINECONE_API_KEY=<Your pine cone API key for vector embedding storage>
        PINECONE_INDEX=ai-hackathon-index
   
        ```
   d. Run the following command to vectorise .mdx files in the knowladge base:
   
       ```
       python3 java-agent-embed.py --index
   
       ```
   
    e. Run the application:
    
        ```
        streamlit run app.py
   
        ```
The **Support Ticket Analyzer** tool launches in your browser, fully operational on your local host.

## How to use
In the Support Analyser tool UI:
* Upload the .csv file containing support tickets.
* Click **Analyze**. The process may take a few minutes to complete. After a successful analysis, the results will be displayed in a tabular format.
* You can download the analysis results in a .csv file.

