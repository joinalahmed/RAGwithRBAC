"""
LangChain Azure AI Search Indexer with proper datetime handling.
Implements correct Edm.DateTimeOffset formatting for Azure AI Search compatibility.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime, timezone
import logging
from pathlib import Path
import json
from dotenv import load_dotenv

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
import json
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_datetime_for_azure_search(dt: datetime) -> str:
    """Format datetime for Azure AI Search's Edm.DateTimeOffset field."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

class AzureSearchIndexer:
    """Implements document indexing with proper datetime handling."""
    
    def __init__(self):
        """Initialize the indexer with Azure service configurations."""
        self._load_env_vars()
        
        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
            openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY')
        )
        
        # Define fields with proper datetime configuration
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
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SimpleField(
                name="group_ids",
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
                name="last_update",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True
            )
        ]
        
        # Initialize vector store
        self.vector_store = AzureSearch(
            azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            azure_search_key=os.getenv('AZURE_SEARCH_ADMIN_KEY'),
            index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
            embedding_function=self.embeddings.embed_query,
            fields=fields
        )

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
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    async def add_documents(
        self,
        docs_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        security_groups: Optional[Dict[str, List[str]]] = None
    ):
        """Process and index documents with proper datetime handling."""
        try:
            #read metadata.json to dict
            doc_metadata = json.load(open("documents/metadata.json"))
            for key, value in doc_metadata.items():
                print(key,doc_metadata[key])
            # Load documents
            if os.path.isfile(docs_path):
                loader = TextLoader(docs_path, encoding="utf-8")
                documents = loader.load()
            else:
                loader = DirectoryLoader(
                    docs_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                documents = loader.load()
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split documents
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            docs = text_splitter.split_documents(documents)
            
            # Process documents with security groups and proper datetime
            if security_groups:
                for doc in docs:
                    rel_path = str(Path(doc.metadata["source"]).relative_to(docs_path))
                    doc_groups = []
                    
                    # Assign security groups
                    doc_meta = doc_metadata[f'documents/{rel_path}']
                    doc_groups = doc_meta["groups"]
                    print(rel_path,doc_groups)
                    '''for pattern, groups in security_groups.items():
                        if pattern in rel_path:
                            doc_groups.extend(groups)
                    
                    if not doc_groups:
                        doc_groups = ["public"]'''
                    
                    # Update metadata with properly formatted datetime
                    doc.metadata.update({
                        "group_ids": doc_groups,
                        "last_update": format_datetime_for_azure_search(datetime.now(timezone.utc)),
                        "title": Path(rel_path).stem
                    })
            
            # Index documents
            await self.vector_store.aadd_documents(documents=docs)
            logger.info(f"Indexed {len(docs)} document chunks")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    async def search_documents(
        self,
        query: str,
        user_groups: Optional[List[str]] = None,
        k: int = 3,
        search_type: str = "hybrid"
    ):
        """Search documents with security filtering."""
        try:
            # Create security filter
            filter_expr = None
            if user_groups:
                filter_expr = f"group_ids/any(g: search.in(g, '{','.join(user_groups)}'))"
            
            # Perform search
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filters=filter_expr,
                search_type=search_type
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

async def main():
    """Run the indexing and search demonstration."""
    try:
        # Initialize indexer
        indexer = AzureSearchIndexer()
        
        # Define security groups
        security_groups = {
            "hr/": ["hr_group", "management"],
            "finance/": ["finance_group", "management"],
            "public": ["all_employees","hr_group","finance_group","management"],
            "confidential": ["management"],
            "restricted": ["management"]
        }
        
        # Index documents
        await indexer.add_documents(
            docs_path="documents",
            chunk_size=500,
            chunk_overlap=100,
            security_groups=security_groups
        )
        
        # Example searches
        queries = [
            ("What are our HR policies?", ["hr_group"]),
            ("Show me public announcements", ["all_employees"]),
            ("What are our strategic plans?", ["management"])
        ]
        
        for query, groups in queries:
            results = await indexer.search_documents(
                query=query,
                user_groups=groups,
                k=3,
                search_type="hybrid"
            )
            
            print(f"\nQuery: {query}")
            print(f"User Groups: {groups}")
            for doc in results:
                print(f"- {doc.page_content[:100]}...")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())