# support_analyser
The Support Analysis Tool is an AI-powered solution that automates the analysis of support tickets, making Global Technical Support more efficient and responsive.

## Overview
You can streamline the analysis process by processing support tickets in batches. The tool automates tasks like ticket categorization and root cause analysis, helping you deliver precise solutions faster and improving both customer satisfaction and team capacity.

## Pre-requisites
* A .csv file with support tickets
* An API key to connect to a Large Language Model (LLM)

## Installation Instructions

1. Clone the [repository](https://github.com/CBehera5/support_analyzer) to your local machine.
2. On your local machine, open the cloned folder in Visual Studio Code.
3. In the terminal, perform the following steps:
    a. Create a virtual environment:
            ```
            python3 -m venv venv
            source venv/bin/activate  
            ```
    b. Install dependencies:
            ```
            pip install -r requirements.txt
            ```
    c. Run the applications:
        ```
        streamlit run app.py
        ```

## How to use

* In the left pane, enter the API key.
* Upload the .csv file containing support tickets.
* Click **Analyze**. The process may take a few minutes to complete. After a successful analysis, the results will be displayed in a tabular format.
* You can download the analysis results in a .csv file.

## What's in the result file

From the result file, you will obtain:

* **Hierarchical Categorization**: Relevant product areas and sub-categories of the issue.
* **Root cause**: underlying cause of the issue if the following format:
    * Primary_Cause_Type
    * Cause_Justification
    * Documentation Issue/UI/UX Issue/Pricing Issue/other
    * Details, if it is an other Product Issue
