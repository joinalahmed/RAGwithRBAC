"""
Quart application implementing a RAG-powered document search system with Azure OpenAI integration.
Includes regional access control and comprehensive security filtering.
"""

from quart import Quart, render_template, request, jsonify
from datetime import datetime, timezone
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
import json
from dotenv import load_dotenv
import asyncio
from openai import AzureOpenAI

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

from azure.search.documents.indexes.models import (
    SearchableField,
    SimpleField,
    SearchField,
    SearchFieldDataType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Quart app
app = Quart(__name__)

def format_datetime_for_azure_search(dt: datetime) -> str:
    """Format datetime for Azure AI Search's Edm.DateTimeOffset field."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

class EnhancedAzureSearchService:
    """Implements secure document search with RAG capabilities and regional access control."""
    
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
        
        # Define regions and access levels
        self.regions = ["us", "uk", "india", "global"]
        self.access_levels = ["public", "internal", "confidential", "restricted", "classified"]

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
        """Initialize Azure services with enhanced field configuration."""
        try:
            # Initialize embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
                openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_KEY')
            )
            
            # Define enhanced fields
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="myHnswProfile"
                ),
                SimpleField(
                    name="group_ids",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    retrievable=True
                ),
                SimpleField(
                    name="regions",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True,
                    retrievable=True
                ),
                SimpleField(
                    name="source",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    retrievable=True
                ),
                SearchableField(
                    name="title",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True
                ),
                SimpleField(
                    name="access_level",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    retrievable=True
                ),
                SimpleField(
                    name="last_update",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                )
            ]
            
            # Initialize vector store with enhanced fields
            self.vector_store = AzureSearch(
                azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
                azure_search_key=os.getenv('AZURE_SEARCH_ADMIN_KEY'),
                index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
                embedding_function=self.embeddings.embed_query,
                fields=fields
            )
            
            # Initialize Azure OpenAI client
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            logger.info("Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {str(e)}")
            raise

    def _expand_user_groups(self, user_groups: List[str]) -> List[str]:
        """Expand user groups based on hierarchical access."""
        expanded = set()
        for group in user_groups:
            print(group)
            if group == 'management':
                expanded.update(['management', 'all_employees', 'hr_group', 'finance_group'])
            elif group in ['hr_group', 'finance_group']:
                expanded.update([group, 'all_employees'])
            else:
                expanded.add(group)
        print(expanded)
        return list(expanded)

    def _get_allowed_access_levels(self, user_groups: List[str]) -> List[str]:
        """Determine allowed access levels based on user groups."""
        print(user_groups)
        if 'management' in user_groups:
            return self.access_levels  # All levels
        elif 'hr_group' in user_groups or 'finance_group' in user_groups:
            return self.access_levels[:4]  # Up to restricted
        elif 'all_employees' in user_groups:
            return self.access_levels[:2]  # Only public and internal
        return [self.access_levels[0]]  # Public only

    async def search_documents(
        self,
        query: str,
        user_groups: List[str],
        regions: Optional[List[str]] = None,
        access_levels: Optional[List[str]] = None,
        full_text: bool = False,
        k: int = 5
    ) -> List[Dict]:
        """Enhanced document search with security and regional filtering."""
        try:
            filter_conditions = []
            
            # Handle group access
            expanded_groups = self._expand_user_groups(user_groups)
            filter_conditions.append(
                f"group_ids/any(g: search.in(g, '{','.join(expanded_groups)}'))"
            )
            
            # Add regional access filter
            if 'global' in regions:
                regions=['india']
            if regions:
                filter_conditions.append(
                    f"regions/any(r: search.in(r, '{','.join(regions)}'))"
                )
            
            # Get allowed access levels and filter with requested levels
            allowed_levels = self._get_allowed_access_levels(expanded_groups)
            print(allowed_levels)
            if access_levels:
                effective_levels = [level for level in access_levels if level in allowed_levels]
            else:
                effective_levels = allowed_levels
                
            if allowed_levels:
                filter_conditions.append(
                    f"search.in(access_level, '{','.join(allowed_levels)}')"
                )
            
            # Combine filters
            filter_str = " and ".join(filter_conditions)
            print(filter_str)
            
            # Determine search type
            
            # Perform search
            results = await self.vector_store.asimilarity_search_with_relevance_scores(
                query,
                k=k,
                filters=filter_str
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                if score >= 0.81:
                    formatted_results.append({
                        'title': doc.metadata.get('title', 'Untitled'),
                        'content': doc.page_content,
                        'score': score,
                        'group_ids': doc.metadata.get('group_ids', []),
                        'regions': doc.metadata.get('regions', []),
                    'access_level': doc.metadata.get('access_level', 'public'),
                    'last_update': doc.metadata.get('last_update',
                        format_datetime_for_azure_search(datetime.now(timezone.utc)))
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def generate_rag_answer(self, query: str, context: str, sources: List[Dict]) -> Dict:
        """Generate AI answer using RAG approach with enhanced context handling."""
        try:
            system_prompt = """You are an AI assistant at Contoso, designed to provide accurate and helpful answers to employee queries related to company policies, plans, HR guidelines, and other internal matters. 

            Guidelines:
            * Focus on relevant, accurate information based solely on the context
            * Maintain clarity and respect confidentiality
            * Use appropriate formatting (paragraphs for descriptions, tables for lists)
            * If context is not relevant, acknowledge limitations
            * Do not reveal sensitive information
            * Stay focused on the query at hand
            * give results in markdown format
            """

            user_prompt = f"""
            Question: {query}

            Information: {context}

            Provide a comprehensive answer following Contoso's guidelines.
            """

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                n=1,
                seed=42
            )
            print(response)
            return {
                'answer': response.choices[0].message.content,
                'token_count': response.usage.total_tokens,
                'sources': sources[:3]
            }
            
        except Exception as e:
            logger.error(f"RAG generation error: {str(e)}")
            raise

    async def upload_document(
        self,
        file_data: bytes,
        filename: str,
        group_ids: List[str],
        region: str = "global",
        access_level: str = "public"
    ) -> Dict:
        """Enhanced document upload with regional and access control."""
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
            
            # Handle regional access
            regions = [region]
            if region == "global":
                regions.extend(["us", "uk", "india"])
            
            # Add enhanced metadata
            for doc in docs:
                doc.metadata.update({
                    'group_ids': group_ids,
                    'regions': regions,
                    'access_level': access_level,
                    'title': filename,
                    'last_update': format_datetime_for_azure_search(datetime.now(timezone.utc))
                })
            
            # Index documents
            await self.vector_store.aadd_documents(documents=docs)
            
            return {
                'status': 'success',
                'chunks_indexed': len(docs),
                'filename': filename,
                'region': region,
                'access_level': access_level
            }
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

# Initialize search service
try:
    search_service = EnhancedAzureSearchService()
    logger.info("Search service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize search service: {str(e)}")
    raise

@app.route('/')
async def index():
    """Render the main search interface."""
    return await render_template('index.html',
                           user_groups=search_service.user_groups,
                           regions=search_service.regions,
                           access_levels=search_service.access_levels)

@app.route('/search', methods=['POST'])
async def search():
    """Handle search requests with RAG support."""
    try:
        data = await request.get_json()
        print(data)
        query = data.get('search_query')
        user_group = data.get('group_id')
        region = data.get('region', 'global')
        access_level = data.get('access_level', ['public'])
        full_text = data.get('full_text', False)
        print(query, user_group, region, access_level, full_text)
        if not query or not user_group:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        results = await search_service.search_documents(
            query=query,
            user_groups=[user_group],
            regions=[region],
            access_levels=access_level,
            full_text=full_text
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
        data = await request.get_json()
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
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"RAG generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate answer'}), 500

@app.route('/upload', methods=['POST'])
async def upload():
    """Handle document uploads."""
    try:
        files = await request.files
        form = await request.form
        
        if 'document' not in files:
            return jsonify({'error': 'No document provided'}), 400
            
        file = files['document']
        group = form.get('group')
        region = form.get('region', 'global')
        access_level = form.get('access_level', 'public')
        
        if not file or not group:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Process and index the document
        result = await search_service.upload_document(
            file_data=await file.read(),
            filename=file.filename,
            group_ids=[group],
            region=region,
            access_level=access_level
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/regenerate_answer', methods=['POST'])
async def regenerate_answer():
    """Regenerate RAG answer with different parameters."""
    try:
        data = await request.get_json()
        query = data.get('query')
        user_group = data.get('group_id', 'all_employees')
        region = data.get('region', 'global')
        access_level = data.get('access_level', ['public'])
        
        if not query:
            return jsonify({'error': 'Missing query parameter'}), 400
            
        # Perform new search and generate answer
        results = await search_service.search_documents(
            query=query,
            user_groups=[user_group],
            regions=[region],
            access_levels=access_level
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

@app.errorhandler(Exception)
async def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(error)
    }), 500

def create_app():
    """Application factory function."""
    # Ensure required directories exist
    Path("temp").mkdir(exist_ok=True)
    
    return app

if __name__ == '__main__':
    # Add startup logging
    logger.info("Starting Quart application...")
    logger.info(f"Environment variables loaded: {list(os.environ.keys())}")
    logger.info(f"Available regions: {search_service.regions}")
    logger.info(f"Available user groups: {list(search_service.user_groups.keys())}")
    
    app.run(debug=True)