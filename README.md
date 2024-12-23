# Azure AI Search Security Filters Demo with RAG Integration

This demo application showcases how to implement document-level security in Azure AI Search using security filters. It demonstrates a practical implementation of secure document search combined with Retrieval Augmented Generation (RAG) using Azure OpenAI.

## Overview

Azure AI Search doesn't provide native document-level permissions but offers powerful filtering capabilities that can be used to implement security trimming. This application demonstrates:

1. **Security Filter Pattern**: Implementation of document-level security using `search.in()` filters
2. **Group-Based Access**: Documents tagged with group identifiers for access control
3. **Regional Filtering**: Multi-region document access management
4. **Hierarchical Security**: Role-based access levels from public to classified
5. **RAG Integration**: Secure document context for AI-generated responses

### Security Implementation
The demo uses the following security filter pattern:
```python
filter = "group_ids/any(g:search.in(g, 'group1,group2')) and regions/any(r:search.in(r, 'us,global'))"
```

Each document in the index includes:
- Group IDs for authorized access
- Regional access controls
- Security classification level

Example document metadata:
```json
{
    "group_ids": ["hr_group", "finance_group"],
    "regions": ["us", "uk", "india", "global"],
    "access_level": "restricted",
    "content": "Document content...",
    "last_update": "2024-01-01T00:00:00Z"
}
```

## Prerequisites

### Required Services
- Azure OpenAI service
- Azure AI Search service
- Azure Storage Account

### Environment Variables
```env
# Azure Search Configuration
AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_ADMIN_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=your_index_name

# Azure OpenAI Configuration
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
AZURE_OPENAI_DEPLOYMENT=your_deployment
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt

# Additional dependencies for document generation and indexing
pip install faker python-lorem azure-storage-blob
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add the required variables (see Prerequisites section)

## Running the Application

1. Start the server:
```bash
python app.py
```

2. Access the web interface:
   - Open browser to `http://localhost:5000`
   - Select a user profile to begin
   - Start searching documents

## Security Model

### Access Levels
1. `public`: Available to all authenticated users
2. `internal`: General internal documents
3. `confidential`: Sensitive information
4. `restricted`: Higher sensitivity
5. `classified`: Highest level of security

### User Groups
- `management`: Full access (all levels)
- `hr_group`: Up to restricted level
- `finance_group`: Up to restricted level
- `all_employees`: Public and internal only

### Document Security Fields
```json
{
    "group_ids": ["hr_group", "finance_group"],
    "regions": ["us", "uk", "india", "global"],
    "access_level": "restricted"
}
```

## Data Generation and Indexing

### Document Generation (doc_gen.py)
This script helps generate dummy documents with proper security constraints and metadata for testing.

```bash
python doc_gen.py --output docs/ --count 100
```

#### Parameters:
- `--output`: Output directory for generated documents (default: docs/)
- `--count`: Number of documents to generate (default: 100)
- `--seed`: Random seed for reproducible generation (optional)

Generated documents include:
```json
{
    "title": "string",
    "content": "string",
    "metadata": {
        "group_ids": ["array of allowed groups"],
        "regions": ["array of allowed regions"],
        "access_level": "string",
        "last_update": "datetime"
    }
}
```

### Document Indexing (index.py)
This script handles the indexing of documents into Azure AI Search with proper security fields.

```bash
python index.py --input docs/ --index gptkbindex
```

#### Parameters:
- `--input`: Directory containing documents to index
- `--index`: Name of the search index
- `--clear`: Clear existing index before indexing (optional)
- `--batch-size`: Number of documents to index in each batch (default: 50)

The indexing process:
1. Creates/updates index schema if needed
2. Processes documents in batches
3. Adds security metadata and embeddings
4. Uploads to Azure AI Search

#### Index Schema:
```json
{
    "name": "gptkbindex",
    "fields": [
        {"name": "id", "type": "Edm.String", "key": true},
        {"name": "content", "type": "Edm.String", "searchable": true},
        {"name": "group_ids", "type": "Collection(Edm.String)", "filterable": true},
        {"name": "regions", "type": "Collection(Edm.String)", "filterable": true},
        {"name": "access_level", "type": "Edm.String", "filterable": true},
        {"name": "content_vector", "type": "Collection(Edm.Single)", "dimensions": 1536, "vectorSearchProfile": "myHnswProfile"}
    ]
}
```

## API Endpoints

### Search
```http
POST /search
Content-Type: application/json

{
    "search_query": "string",
    "group_id": "string",
    "region": "string",
    "access_level": ["string"],
    "full_text": boolean
}
```

### Generate RAG Answer
```http
POST /generate_answer
Content-Type: application/json

{
    "query": "string",
    "context": "string",
    "sources": [{"title": "string", "score": number}]
}
```

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

document: file
group: string
region: string
access_level: string
```


## Troubleshooting

### Common Issues

1. **Search Returns No Results**
   - Check user's group permissions
   - Verify document access levels
   - Confirm regional access settings

2. **Upload Failures**
   - Verify storage account access
   - Check file permissions
   - Confirm correct group/region settings

3. **RAG Generation Issues**
   - Verify Azure OpenAI quota and limits
   - Check document context length
   - Confirm API credentials

## Future Enhancements

The following features are planned for future releases:

1. **Microsoft Entra ID Integration**
   - Native authentication/authorization
   - Group synchronization
   - Token-based access

2. **Enhanced Security**
   - Role-based access control (RBAC)
   - Audit logging
   - Advanced permission management
