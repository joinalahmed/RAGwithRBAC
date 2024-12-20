import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Set up logging with a detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentGenerator:
    """
    Generates sample documents with role-based access control using Azure OpenAI.
    Implements security classifications and user group permissions for document access.
    """
    
    def __init__(self):
        """
        Initialize the document generator with Azure OpenAI configuration and document templates.
        Defines document categories, access levels, and user groups.
        """
        # Load and validate environment variables
        self._load_env_vars()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Define user groups and their access levels
        self.user_groups = {
            "all_employees": "Basic access for all employees",
            "hr_group": "HR team members",
            "finance_group": "Finance team members",
            "management": "Senior management and executives"
        }
        
        # Define document structure with granular access controls
        self.document_types = {
            "hr": {
                "path": "documents/hr",
                "public_content": [
                    {
                        "name": "leave_policy.txt",
                        "groups": ["all_employees"],
                        "system_prompt": """You are an HR policy writer creating clear, 
                        accessible content for all employees.""",
                        "user_prompt": """Create a comprehensive leave policy document covering:
                        1. Types of leave (vacation, sick, parental, bereavement)
                        2. Leaves employees get in each category
                        3. Leave request and approval process
                        4. Holiday calendar and time-off scheduling
                        5. Return-to-work procedures
                        6. Emergency leave protocols
                        
                        Use clear, simple language that all employees can understand."""
                    },
                    {
                        "name": "workplace_safety.txt",
                        "groups": ["all_employees"],
                        "system_prompt": """You are a workplace safety expert creating 
                        guidelines for all employees.""",
                        "user_prompt": """Create safety guidelines covering:
                        1. Emergency procedures and evacuation plans
                        2. Workplace health and safety practices
                        3. Incident reporting procedures
                        4. First aid and medical emergency response
                        5. Remote work safety guidelines
                        
                        Focus on practical, actionable information."""
                    }
                ],
                "restricted_content": [
                    {
                        "name": "compensation_guidelines.txt",
                        "groups": ["hr_group", "management"],
                        "system_prompt": """You are creating confidential HR documentation 
                        for management and HR personnel.""",
                        "user_prompt": """Create detailed compensation guidelines covering:
                        1. Salary bands and structures
                        2. Performance-based compensation
                        3. Bonus calculation methods
                        4. Equity compensation policies
                        5. Executive compensation framework
                        
                        Include specific formulas and confidential procedures."""
                    },
                    {
                        "name": "personnel_records_policies.txt",
                        "groups": ["hr_group"],
                        "system_prompt": """You are creating sensitive HR documentation 
                        for internal HR use only.""",
                        "user_prompt": """Create guidelines for handling personnel records:
                        1. Employee data privacy requirements
                        2. Record retention policies
                        3. Access control procedures
                        4. Data breach response protocols
                        5. Compliance requirements
                        
                        Include specific security measures and handling procedures."""
                    },
                    {
                        "name": "personnel_records.txt",
                        "groups": ["hr_group"],
                        "system_prompt": """You are creating sensitive employee records 
                        for internal HR use only.""",
                        "user_prompt": """Create personnel records:
                        1. Employee name, address, phone number, email, and social security number
                        2. Employee salary and benefits information
                        3. Employee performance reviews and evaluations
                        4. Employee disciplinary records
                        5. Employee training and development records"""
                    }
                ]
            },
            "finance": {
                "path": "documents/finance",
                "public_content": [
                    {
                        "name": "expense_guidelines.txt",
                        "groups": ["all_employees"],
                        "system_prompt": """You are creating clear financial guidelines 
                        for all employees.""",
                        "user_prompt": """Create expense reporting guidelines covering:
                        1. Eligible expenses and limits
                        2. Submission process and deadlines
                        3. Required documentation
                        4. Reimbursement timeline
                        5. Travel expense policies
                        
                        Use clear examples and specific procedures."""
                    }
                ],
                "restricted_content": [
                    {
                        "name": "financial_forecasts.txt",
                        "groups": ["finance_group", "management"],
                        "system_prompt": """You are creating confidential financial 
                        documentation for finance team and management.""",
                        "user_prompt": """Create detailed financial forecasts covering:
                        1. Revenue projections and assumptions
                        2. Cost structure analysis
                        3. Investment strategies
                        4. Risk assessments
                        5. Market analysis
                        
                        Include specific financial data and analysis."""
                    }

                ]
            },
            "operations": {
                "path": "documents/operations",
                "public_content": [
                    {
                        "name": "company_handbook.txt",
                        "groups": ["all_employees"],
                        "system_prompt": """You are creating clear operational guidelines 
                        for all employees.""",
                        "user_prompt": """Create a company handbook covering:
                        1. Company values and culture
                        2. General workplace policies
                        3. Communication guidelines
                        4. IT usage policies
                        5. Professional development opportunities
                        
                        Focus on information relevant to all employees."""
                    }
                ],
                "restricted_content": [
                    {
                        "name": "strategic_plans.txt",
                        "groups": ["management"],
                        "system_prompt": """You are creating confidential strategic 
                        documentation for senior management.""",
                        "user_prompt": """Create strategic planning documentation covering:
                        1. Growth strategies and objectives
                        2. Market expansion plans
                        3. Competitive analysis
                        4. Resource allocation strategy
                        5. Risk mitigation plans
                        
                        Include confidential strategic information."""
                    },
                    {
                        "name": "expansion_plans.txt",
                        "groups": ["management"],
                        "system_prompt": """You are creating confidential expansion plans   
                        for senior management.""",
                        "user_prompt": """Create expansion planning documentation covering:
                        1. Expansion strategies and objectives
                        2. Market expansion plans
                        3. Competitive analysis
                        4. Resource allocation strategy
                        5. Risk mitigation plans
                        
                        Include confidential expansion information."""
                    },
                    {
                        "name": "acquisition_plans  .txt",
                        "groups": [ "management"],
                        "system_prompt": """You are creating confidential acquisition plans 
                        for senior management.""",
                        "user_prompt": """Create acquisition planning documentation covering:
                        1. Acquisition strategies and objectives
                        2. Target acquisition criteria
                        3. Competitive analysis
                        3. Investment strategies
                        4. Risk assessments
                        5. Market analysis
                        
                        Include specific financial data and analysis."""
                    }
                ]
            }
        }
    
    def _load_env_vars(self):
        """
        Load and validate required environment variables for Azure OpenAI configuration.
        """
        load_dotenv()
        
        required_vars = [
            'AZURE_OPENAI_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_API_VERSION',
            'AZURE_OPENAI_DEPLOYMENT'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    async def generate_document_content(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate document content using Azure OpenAI with specific prompts.
        
        Args:
            system_prompt: Context and role information for the AI
            user_prompt: Specific content generation instructions
            
        Returns:
            Generated document content as a string
        """
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2500,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
    
    def create_directory_structure(self):
        """
        Create the hierarchical directory structure for document organization.
        """
        for doc_type, info in self.document_types.items():
            Path(info["path"]).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {info['path']}")
    
    async def generate_all_documents(self) -> Dict:
        """
        Generate all documents with appropriate access controls and metadata.
        
        Returns:
            Dictionary containing metadata for all generated documents
        """
        self.create_directory_structure()
        metadata = {}
        
        for doc_type, info in self.document_types.items():
            logger.info(f"Generating {doc_type} documents...")
            
            # Generate public content
            for template in info["public_content"]:
                file_path = Path(info["path"]) / template["name"]
                try:
                    content = await self.generate_document_content(
                        template["system_prompt"],
                        template["user_prompt"]
                    )
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    metadata[str(file_path)] = {
                        "type": doc_type,
                        "access_level": "public",
                        "groups": template["groups"],
                        "created": datetime.utcnow().isoformat(),
                        "title": template["name"].replace(".txt", "").replace("_", " ").title(),
                        "last_modified": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Generated public document: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error generating {file_path}: {str(e)}")
                    continue
            
            # Generate restricted content
            for template in info["restricted_content"]:
                file_path = Path(info["path"]) / template["name"]
                try:
                    content = await self.generate_document_content(
                        template["system_prompt"],
                        template["user_prompt"]
                    )
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    metadata[str(file_path)] = {
                        "type": doc_type,
                        "access_level": "restricted",
                        "groups": template["groups"],
                        "created": datetime.utcnow().isoformat(),
                        "title": template["name"].replace(".txt", "").replace("_", " ").title(),
                        "last_modified": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Generated restricted document: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error generating {file_path}: {str(e)}")
                    continue
        
        # Save metadata with access control information
        metadata_path = Path("documents/metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Document generation completed successfully")
        return metadata

async def main():
    """
    Main function to orchestrate the document generation process.
    Provides a summary of generated content and access controls.
    """
    try:
        generator = DocumentGenerator()
        metadata = await generator.generate_all_documents()
        
        # Print detailed summary of generated documents
        print("\nDocument Generation Summary:")
        print("=" * 50)
        
        for doc_type in generator.document_types:
            print(f"\n{doc_type.upper()}:")
            print("-" * 30)
            
            # Count public documents
            public_docs = sum(1 for m in metadata.values() 
                            if m["type"] == doc_type and m["access_level"] == "public")
            print(f"Public documents: {public_docs}")
            
            # Count restricted documents
            restricted_docs = sum(1 for m in metadata.values() 
                                if m["type"] == doc_type and m["access_level"] == "restricted")
            print(f"Restricted documents: {restricted_docs}")
            
            # Show access patterns
            print("\nAccess Patterns:")
            for doc_path, meta in metadata.items():
                if meta["type"] == doc_type:
                    print(f"  - {Path(doc_path).name}")
                    print(f"    Access Level: {meta['access_level']}")
                    print(f"    Groups: {', '.join(meta['groups'])}")
            
    except Exception as e:
        logger.error(f"Document generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())