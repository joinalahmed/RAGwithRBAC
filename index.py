"""
LangChain Azure AI Search Indexer with proper datetime handling and regional access control.
Implements correct Edm.DateTimeOffset formatting and region-based filtering for Azure AI Search.
"""

import os
from typing import List, Dict, Optional, Union
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_datetime_for_azure_search(dt: datetime) -> str:
    """Format datetime for Azure AI Search's Edm.DateTimeOffset field."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

class AzureSearchIndexer:
    """Implements document indexing with proper datetime handling and regional access control."""
    
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
        
        # Define fields with proper datetime configuration and regional access
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
        """Process and index documents with proper datetime handling and regional access."""
        try:
            # Read metadata.json to dict
            doc_metadata = json.load(open("documents/metadata.json"))
            
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
            
            # Process documents with security groups and regional access
            for doc in docs:
                rel_path = str(Path(doc.metadata["source"]).relative_to(docs_path))
                full_path = f'documents/{rel_path}'
                
                if full_path in doc_metadata:
                    doc_meta = doc_metadata[full_path]
                    
                    # Get document metadata
                    doc_groups = doc_meta.get("groups", [])
                    doc_region = doc_meta.get("region", "global")
                    access_level = doc_meta.get("access_level", "public")
                    
                    # Handle regional access
                    regions = [doc_region]
                    if doc_region == "global":
                        regions.extend(["us", "uk", "india"])  # Global docs accessible in all regions
                    
                    # Update metadata with security and regional information
                    doc.metadata.update({
                        "group_ids": doc_groups,
                        "regions": regions,
                        "access_level": access_level,
                        "last_update": format_datetime_for_azure_search(datetime.now(timezone.utc)),
                        "title": Path(rel_path).stem
                    })
                else:
                    logger.warning(f"No metadata found for document: {full_path}")
                    doc.metadata.update({
                        "group_ids": ["all_employees"],
                        "regions": ["global"],
                        "access_level": "public",
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
        regions: Optional[List[str]] = None,
        access_levels: Optional[List[str]] = None,
        k: int = 3,
        search_type: str = "hybrid"
    ):
        """Search documents with security and regional filtering."""
        try:
            filter_conditions = []
            
            # Add security group filter
            if user_groups:
                filter_conditions.append(
                    f"group_ids/any(g: search.in(g, '{','.join(user_groups)}'))"
                )
            
            # Add regional access filter
            if regions:
                filter_conditions.append(
                    f"regions/any(r: search.in(r, '{','.join(regions)}'))"
                )
            
            # Add access level filter
            if access_levels:
                filter_conditions.append(
                    f"search.in(access_level, '{','.join(access_levels)}')"
                )
            
            # Combine all filters
            filter_expr = None
            if filter_conditions:
                filter_expr = " and ".join(filter_conditions)
            
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
        
        # Index documents
        await indexer.add_documents(
            docs_path="documents",
            chunk_size=1000,
            chunk_overlap=400
        )
        
        # Example searches with different access patterns
        test_queries = [
            {
                "query": "how many leaves do i get?",
                "user_groups": ["all_employees"],
                "regions": ["india"],
                "access_levels": ["public"]
            }
        ]
        
        for test in test_queries:
            results = await indexer.search_documents(
                query=test["query"],
                user_groups=test["user_groups"],
                regions=test["regions"],
                access_levels=test["access_levels"],
                k=3,
                search_type="hybrid"
            )
            
            print(f"\nQuery: {test['query']}")
            print(f"User Groups: {test['user_groups']}")
            print(f"Regions: {test['regions']}")
            print(f"Access Levels: {test['access_levels']}")
            print("\nResults:")
            for doc in results:
                print(f"\nTitle: {doc.metadata.get('title', 'N/A')}")
                print(f"Region: {doc.metadata.get('regions', ['N/A'])[0]}")
                print(f"Access Level: {doc.metadata.get('access_level', 'N/A')}")
                print(f"Content Preview: {doc.page_content[:200]}...")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())