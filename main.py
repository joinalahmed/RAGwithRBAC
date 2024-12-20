"""
Flask application implementing a RAG-powered document search system with Azure OpenAI integration.
This application provides secure document search with AI-generated summaries and proper access control.
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
import json
from dotenv import load_dotenv
import asyncio
from openai import AzureOpenAI

# Import necessary Azure and LangChain components
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class AzureSearchService:
    """Implements secure document search with RAG capabilities."""
    
    def __init__(self):
        """Initialize Azure services and configure search components."""
        self._load_env_vars()
        self._initialize_azure_services()
        
        # Define available user groups with friendly names
        self.user_groups = {
            "hr_group": "HR Team",
            "finance_group": "Finance Team",
            "management": "Management",
            "all_employees": "All Employees"
        }

    def _load_env_vars(self):
        """Load and validate environment variables."""
        load_dotenv()
        
        required_vars = [
            'AZURE_SEARCH_ENDPOINT',
            'AZURE_SEARCH_ADMIN_KEY',
            'AZURE_SEARCH_INDEX_NAME',
            'AZURE_OPENAI_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_API_VERSION',
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
            'AZURE_OPENAI_DEPLOYMENT'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _initialize_azure_services(self):
        """Initialize Azure OpenAI and Search services."""
        try:
            # Initialize embeddings for vector search
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
                openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_KEY')
            )
            
            # Initialize vector store for document search
            self.vector_store = AzureSearch(
                azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
                azure_search_key=os.getenv('AZURE_SEARCH_ADMIN_KEY'),
                index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
                embedding_function=self.embeddings.embed_query
            )
            
            # Initialize Azure OpenAI client for RAG
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            logger.info("Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {str(e)}")
            raise

    async def search_documents(self, query: str, user_groups: List[str], 
                             full_text: bool = False, exact_match: bool = False) -> List[Dict]:
        """
        Search documents with security filtering and ranking.
        
        Args:
            query: Search query text
            user_groups: List of user's security groups
            full_text: Whether to perform full text search
            exact_match: Whether to require exact matches
            
        Returns:
            List of search results with metadata
        """
        try:
            # Create security filter
            if user_groups==['all_employees']:
                user_groups = ['all_employees']
            if user_groups==['management']:
                user_groups = ['management',"all_employees","hr_group","finance_group"]
            if user_groups==['hr_group']:
                user_groups = ['hr_group',"all_employees"]
            if user_groups==['finance_group']:
                user_groups = ['finance_group',"all_employees"]
            filter_str = f"group_ids/any(g: search.in(g, '{','.join(user_groups)}'))"
            print(filter_str)
            # Determine search type based on parameters
            search_type = "similarity"
            if full_text:
                search_type = "hybrid"
            
            # Perform search with proper parameters
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=5,  # Return top 5 results
                filters=filter_str
            )
            
            # Format results for display
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'title': doc.metadata.get('title', 'Untitled'),
                    'content': doc.page_content,
                    'score': score,
                    'group_ids': doc.metadata.get('group_ids', []),
                    'last_update': doc.metadata.get('last_update', 
                        datetime.now(timezone.utc).isoformat())
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def generate_rag_answer(self, query: str, context: str, 
                                sources: List[Dict]) -> Dict:
        """
        Generate an AI answer using RAG approach.
        
        Args:
            query: User's question
            context: Retrieved document contexts
            sources: Source documents with metadata
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            # Create a comprehensive prompt for the RAG response
            system_prompt = """You are an AI assistant at Contoso, designed to provide accurate and helpful answers to employee queries related to company policies, plans, HR guidelines, and other internal matters. 

            **Task:**
            Given a query and relevant context from Contoso's internal documents, provide a concise and informative response.

            **Guidelines:**
            * **Focus on Relevance:** Ensure your answer directly addresses the query and is based solely on the provided context.
            * **Clarity and Detail:** Present information clearly and avoid unnecessary details.
            * **Accuracy:** Verify information against the context to maintain correctness.
            * **Confidentiality:** Respect privacy by avoiding disclosure of sensitive information.
            * **Multiple Perspectives:** If multiple sources offer differing viewpoints, synthesize them into a balanced and objective response.
            * **Concise Formatting:** Use clear formatting (e.g., bullet points, tables) to enhance readability.
            * **Direct and Informative:** Avoid unnecessary conversational elements and focus on providing the requested information.
            * **Answer in paragraphs and tables:** If the answer is a list of items, present it as a table with columns for each item.
            * if the context is not relevant to the question, say so and respond i don't know. do not use words like context or documents. You do not have any personal info about anyone.
            * if the question is not related to contoso, say so and respond i don't know. do not use words like context or documents. You do not have any personal info about anyone.
            """

            user_prompt = f"""
            **Query:** {query}

            **Context:** {context} 

            Please provide a comprehensive answer based on the given context, adhering to Contoso's policies and guidelines. 
            """
            # Generate response using Azure OpenAI
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=400,
                n=1,
                seed=42
            )

            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'token_count': response.usage.total_tokens,
                'sources': sources[:3]  # Include top 3 sources
            }
            
        except Exception as e:
            logger.error(f"RAG generation error: {str(e)}")
            raise

    async def upload_document(self, file_data: bytes, filename: str, 
                            group_ids: List[str]) -> Dict:
        """
        Handle document upload and indexing.
        
        Args:
            file_data: Document content
            filename: Name of the uploaded file
            group_ids: Security groups for access control
            
        Returns:
            Upload status and metadata
        """
        temp_path = None
        try:
            # Save temporary file
            temp_path = Path("temp") / filename
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(file_data)
            
            # Process document
            loader = TextLoader(str(temp_path))
            documents = loader.load()
            
            # Split text
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'group_ids': group_ids,
                    'title': filename,
                    'last_update': datetime.now(timezone.utc).isoformat()
                })
            
            # Index documents
            self.vector_store.add_documents(documents=docs)
            
            return {
                'status': 'success',
                'chunks_indexed': len(docs),
                'filename': filename
            }
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

# Initialize search service
try:
    search_service = AzureSearchService()
    logger.info("Search service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize search service: {str(e)}")
    raise

@app.route('/')
def index():
    """Render the main search interface."""
    return render_template('index.html', user_groups=search_service.user_groups)

@app.route('/search', methods=['POST'])
async def search():
    """Handle search requests with RAG support."""
    try:
        data = request.get_json()
        query = data.get('query')
        user_group = data.get('user_group')
        full_text = data.get('full_text', False)
        exact_match = data.get('exact_match', False)
        
        if not query or not user_group:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Perform document search
        results = await search_service.search_documents(
            query=query,
            user_groups=[user_group],
            full_text=full_text,
            exact_match=exact_match
        )
        print(results)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/generate_answer', methods=['POST'])
async def generate_answer():
    """Generate RAG-based answer from search results."""
    try:
        data = request.get_json()
        query = data.get('query')
        context = data.get('context')
        sources = data.get('sources')
        
        if not all([query, context, sources]):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Generate RAG answer
        result = await search_service.generate_rag_answer(
            query=query,
            context=context,
            sources=sources
        )
        print(result)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"RAG generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate answer'}), 500

@app.route('/upload', methods=['POST'])
async def upload():
    """Handle document uploads."""
    try:
        if 'document' not in request.files:
            return jsonify({'error': 'No document provided'}), 400
            
        file = request.files['document']
        group = request.form.get('group')
        
        if not file or not group:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Process and index the document
        result = await search_service.upload_document(
            file_data=file.read(),
            filename=file.filename,
            group_ids=[group]
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/regenerate_answer', methods=['POST'])
async def regenerate_answer():
    """Regenerate RAG answer with different parameters."""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Missing query parameter'}), 400
            
        # Perform new search and generate answer
        results = await search_service.search_documents(
            query=query,
            user_groups=['all_employees']  # Default to all employees for regeneration
        )
        
        if not results:
            return jsonify({'error': 'No relevant documents found'}), 404
            
        context = '\n\n'.join(r['content'] for r in results)
        result = await search_service.generate_rag_answer(
            query=query,
            context=context,
            sources=results
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Answer regeneration error: {str(e)}")
        return jsonify({'error': 'Failed to regenerate answer'}), 500

if __name__ == '__main__':
    # Add more detailed logging for startup
    logger.info("Starting Flask application...")
    logger.info(f"Environment variables loaded: {list(os.environ.keys())}")
    
    app.run(debug=True)