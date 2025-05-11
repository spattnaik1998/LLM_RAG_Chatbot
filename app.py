# app.py
import os
import numpy as np
from typing import List, Dict, Any
import faiss
import openai
from dotenv import load_dotenv
import textwrap
import PyPDF2
import re
from flask import Flask, render_template, request, jsonify

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class Document:
    """A simple document class to store text content and metadata."""
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}

class SarthakRAG:
    """
    A RAG implementation using OpenAI embeddings and FAISS.
    """
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: The OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.documents = []
        self.index = None
        self.embedding_dimension = 1536  # Dimension for OpenAI embeddings
        print(f"Initializing SarthakRAG with embedding model: {embedding_model}")

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of Document objects
        """
        if not documents:
            print("Warning: No documents provided to add_documents")
            return
            
        self.documents.extend(documents)
        
        # Create embeddings for the documents
        print(f"Creating embeddings for {len(documents)} documents...")
        embeddings = self._get_embeddings([doc.text for doc in documents])
        
        if not embeddings:
            print("Error: Failed to generate embeddings")
            return
            
        print(f"Successfully created {len(embeddings)} embeddings")
        
        # Initialize FAISS index if not already done
        if self.index is None:
            print(f"Initializing FAISS index with dimension {self.embedding_dimension}")
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Add embeddings to the index
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            print(f"Added embeddings to FAISS index. Index now contains {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error adding embeddings to FAISS index: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Added {len(documents)} documents. Total documents: {len(self.documents)}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            print("Warning: Empty text list provided to _get_embeddings")
            return []
            
        embeddings = []
        
        # Process in batches to handle API limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                print(f"Getting embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} (size: {len(batch)})")
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                print(f"Successfully created {len(batch_embeddings)} embeddings for batch")
                
            except Exception as e:
                print(f"Error getting embeddings from OpenAI API: {e}")
                import traceback
                traceback.print_exc()
                return []
        
        print(f"Total embeddings created: {len(embeddings)}")
        return embeddings

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search for relevant documents based on the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        if not self.documents or self.index is None:
            print("Warning: No documents or index available for search")
            return []
        
        print(f"Searching for documents relevant to query: '{query}'")
        print(f"Documents available: {len(self.documents)}, Index size: {self.index.ntotal}")
        
        # Get embedding for the query
        query_embedding = self._get_embeddings([query])
        if not query_embedding:
            print("Error: Failed to generate embedding for query")
            return []
            
        query_embedding = query_embedding[0]
        
        # Search in the FAISS index
        try:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), 
                min(top_k, len(self.documents))
            )
            
            # Print search results for debugging
            print(f"Search results: Found {len(indices[0])} documents")
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_preview = doc.text[:50] + "..." if len(doc.text) > 50 else doc.text
                    print(f"  {i+1}. Document index {idx}, distance {dist:.4f}, preview: '{doc_preview}'")
                else:
                    print(f"  {i+1}. Invalid document index: {idx}")
            
            # Return the relevant documents
            result = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
            print(f"Returning {len(result)} relevant documents")
            return result
            
        except Exception as e:
            print(f"Error searching in FAISS index: {e}")
            import traceback
            traceback.print_exc()
            return []

    def query(self, query: str, top_k: int = 3) -> str:
        """
        Perform a RAG query: retrieve documents and generate a response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.search(query, top_k)
        
        if not retrieved_docs:
            # If no documents are found, provide a more helpful response
            return "I don't have specific information about that in Sarthak's resume. Could you try asking something else about Sarthak's skills, experience, or education?"
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            # Don't truncate document text - use the full content
            doc_text = doc.text
            source_info = f"(Source: {doc.metadata.get('source', 'Unknown')}"
            if 'page' in doc.metadata:
                source_info += f", Page {doc.metadata['page']}"
            source_info += ")"
            context_parts.append(f"Document {i+1} {source_info}: {doc_text}")
        
        context = "\n\n".join(context_parts)
        
        # Print debugging information
        print(f"\n--- RAG Query Debug ---")
        print(f"Query: {query}")
        print(f"Number of documents retrieved: {len(retrieved_docs)}")
        print(f"Context length: {len(context)} characters")
        print("First few characters of context:", context[:100] + "...")
        
        # Generate response using GPT-4o with improved system prompt
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "You are 'All about Sarthak', a helpful assistant that answers questions about Sarthak "
                        "based on the provided resume context. Your responses should be personable and conversational. "
                        "IMPORTANT: Always use the provided context to answer questions. Don't make up information about "
                        "Sarthak that isn't supported by the context. If the context doesn't provide enough information "
                        "to answer a question, say that you don't have that specific information in Sarthak's resume. "
                        "Speak in first person as if you are representing Sarthak."
                    )},
                    {"role": "user", "content": f"Context from Sarthak's resume:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Print the generated answer for debugging
            print("\nGenerated answer:", answer[:100] + "..." if len(answer) > 100 else answer)
            print("--- End Debug ---\n")
            
            return answer
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing your question. Please try again or ask something different about Sarthak."

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file and split it into chunks.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    documents = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"Processing PDF: {pdf_path} with {num_pages} pages")
            
            # Process the entire document as one chunk first for better context
            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + " "
            
            # Clean the full text
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            if full_text:
                # Add the entire document as one chunk for better context preservation
                doc = Document(
                    text=full_text,
                    metadata={
                        "source": pdf_path,
                        "type": "full_document"
                    }
                )
                documents.append(doc)
            
            # Now process each page individually
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Clean the text
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Skip empty pages
                if not text:
                    continue
                    
                # Create document with page metadata
                doc = Document(
                    text=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "total_pages": num_pages
                    }
                )
                documents.append(doc)
                
            print(f"Extracted {len(documents)} text chunks from {pdf_path}")
            
            # Print a sample of the first document to verify content extraction
            if documents:
                first_doc = documents[0]
                print(f"Sample text from first chunk ({len(first_doc.text)} chars):")
                print(first_doc.text[:200] + "..." if len(first_doc.text) > 200 else first_doc.text)
                
            return documents
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        import traceback
        traceback.print_exc()
        return []

# Initialize Flask app
app = Flask(__name__)

# Initialize the RAG system
rag = SarthakRAG()

# Define initialization function
def initialize_rag():
    pdf_path = "resume.pdf"
    pdf_documents = extract_text_from_pdf(pdf_path)
    
    if not pdf_documents:
        print(f"Error: Could not extract any text from {pdf_path}.")
        print("Please ensure the PDF file exists and is readable.")
    else:
        rag.add_documents(pdf_documents)
        print("RAG system initialized successfully with resume data.")

# Initialize the RAG system before the first request
# This approach works with newer Flask versions (2.3.0+)
@app.route('/initialize', methods=['GET'])
def init_app():
    initialize_rag()
    return jsonify({"status": "Initialization complete"})

# Call initialization when starting the app
with app.app_context():
    initialize_rag()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"response": "Please provide a question about Sarthak."})
    
    try:
        response = rag.query(query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"response": "Sorry, I encountered an error processing your question. Please try again."})

if __name__ == '__main__':
    app.run(debug=True)