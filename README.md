# LLM_RAG_Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions about Sarthak based on resume information.

# Overview
This application provides an interactive chatbot interface that allows users to ask questions about Sarthak's skills, experience, education, and other resume information. The system uses RAG technology to:

Extract and process text from Sarthak's resume PDF
Find relevant sections that match user queries using vector similarity search
Generate accurate, conversational responses based on the retrieved information

# Features
PDF Text Extraction: Automatically processes resume PDF to extract text content
Vector Search: Uses OpenAI embeddings and FAISS for efficient semantic search
Context-Aware Responses: Retrieves relevant resume sections to provide accurate answers
Conversational UI: Simple web interface for asking questions about Sarthak

# Technical Stack
Backend: Flask web server
Text Processing: PyPDF2 for PDF parsing
Vector Database: FAISS for similarity search
Embeddings: OpenAI text-embedding-3-small model
LLM: GPT-4o for natural language generation
Frontend: HTML/JavaScript (templates directory)

# Setup and Installation
Prerequisites

Python 3.8+
OpenAI API key
Sarthak's resume in PDF format

Environment Setup

Clone the repository
Install dependencies:
pip install -r requirements.txt

Create a .env file with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Place Sarthak's resume as resume.pdf in the project root directory

# Running the Application

Start the Flask server:
python app.py

Access the web interface at http://localhost:5000

# Usage

Navigate to the application in your web browser
Type questions about Sarthak in the chat interface
The system will respond with information from Sarthak's resume

Example questions:

"What are Sarthak's technical skills?"
"Where did Sarthak go to school?"
"What experience does Sarthak have with machine learning?"
"Tell me about Sarthak's work history"

Implementation Details
SarthakRAG Class
The core RAG functionality is implemented in the SarthakRAG class, which:

# Manages document processing and storage
Creates and stores embeddings for resume sections
Performs vector similarity search using FAISS
Integrates with OpenAI for embeddings and response generation

# Document Processing
The system processes Sarthak's resume in two ways:

As a single complete document for comprehensive context
As individual pages for more granular retrieval

Search and Query Process
When a user asks a question:

The query is embedded using OpenAI's embedding model
FAISS performs similarity search to find relevant resume sections
Top matching sections are combined as context
GPT-4o generates a natural, informative response
