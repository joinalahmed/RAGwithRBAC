import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentGenerator:
    def __init__(self):
        self._load_env_vars()
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Regional specifications with comprehensive details
        self.regional_specs = {
            "us": {
                "currency": "USD",
                "bands": {
                    "individual_contributor": [
                        {"level": "IC1", "title": "Associate", "range": (60000, 85000)},
                        {"level": "IC2", "title": "Professional", "range": (80000, 120000)},
                        {"level": "IC3", "title": "Senior", "range": (115000, 170000)},
                        {"level": "IC4", "title": "Principal", "range": (160000, 250000)},
                        {"level": "IC5", "title": "Distinguished", "range": (230000, 400000)},
                        {"level": "IC6", "title": "Fellow", "range": (350000, 600000)}
                    ],
                    "management": [
                        {"level": "M1", "title": "Manager", "range": (130000, 190000)},
                        {"level": "M2", "title": "Senior Manager", "range": (170000, 260000)},
                        {"level": "M3", "title": "Director", "range": (220000, 350000)},
                        {"level": "M4", "title": "Senior Director", "range": (300000, 500000)},
                        {"level": "M5", "title": "VP", "range": (400000, 800000)},
                        {"level": "M6", "title": "SVP", "range": (600000, 1200000)}
                    ]
                },
                "job_families": {
                    "engineering": {
                        "multiplier": 1.1,
                        "roles": ["Software Engineer", "DevOps Engineer", "Security Engineer"],
                        "skills_premium": {
                            "AI/ML": 0.15,
                            "Cloud Architecture": 0.12,
                            "Security": 0.10
                        }
                    },
                    "data_science": {
                        "multiplier": 1.15,
                        "roles": ["Data Scientist", "ML Engineer", "Data Analyst"],
                        "skills_premium": {
                            "Deep Learning": 0.15,
                            "NLP": 0.12,
                            "Computer Vision": 0.12
                        }
                    },
                    "product": {
                        "multiplier": 1.0,
                        "roles": ["Product Manager", "Product Owner", "Business Analyst"],
                        "skills_premium": {
                            "Technical Background": 0.10,
                            "MBA": 0.08
                        }
                    },
                    "design": {
                        "multiplier": 0.95,
                        "roles": ["UX Designer", "UI Designer", "Product Designer"],
                        "skills_premium": {
                            "Design Systems": 0.08,
                            "User Research": 0.07
                        }
                    }
                },
                "leave_policy": {
                    "annual": {
                        "days": 15,
                        "accrual_rate": "1.25 days per month",
                        "carryover_limit": 5
                    },
                    "sick": {
                        "days": 10,
                        "paid": True,
                        "verification_required": "After 3 consecutive days"
                    },
                    "personal": {
                        "days": 5,
                        "restrictions": "48 hours notice required"
                    },
                    "parental": {
                        "primary": {
                            "weeks": 16,
                            "pay": "100% for 12 weeks, 60% for 4 weeks"
                        },
                        "secondary": {
                            "weeks": 6,
                            "pay": "100%"
                        }
                    },
                    "public_holidays": {
                        "count": 11,
                        "floating_holidays": 2
                    }
                },
                "benefits": {
                    "health_insurance": {
                        "coverage": "90%",
                        "providers": ["Blue Cross", "United Healthcare"],
                        "includes_dental": True,
                        "includes_vision": True,
                        "deductible_range": (500, 2500)
                    },
                    "401k_match": {
                        "percentage": "100% up to 6%",
                        "vesting_schedule": "immediate"
                    },
                    "stock_options": {
                        "type": "RSU",
                        "vesting_schedule": "25% per year over 4 years",
                        "cliff": "1 year"
                    },
                    "additional": {
                        "wellness_allowance": 1000,
                        "education_allowance": 5000,
                        "home_office_allowance": 1000
                    }
                },
                "performance_review": {
                    "frequency": "Semi-annual",
                    "bonus_timeline": "Annual",
                    "promotion_cycles": ["January", "July"],
                    "rating_scale": ["Below", "Meets", "Exceeds", "Outstanding"]
                }
            },
            "uk": {
                "currency": "GBP",
                "bands": {
                    "individual_contributor": [
                        {"level": "IC1", "title": "Associate", "range": (35000, 50000)},
                        {"level": "IC2", "title": "Professional", "range": (45000, 75000)},
                        {"level": "IC3", "title": "Senior", "range": (70000, 110000)},
                        {"level": "IC4", "title": "Principal", "range": (100000, 160000)},
                        {"level": "IC5", "title": "Distinguished", "range": (150000, 250000)},
                        {"level": "IC6", "title": "Fellow", "range": (230000, 400000)}
                    ],
                    "management": [
                        {"level": "M1", "title": "Manager", "range": (80000, 120000)},
                        {"level": "M2", "title": "Senior Manager", "range": (110000, 170000)},
                        {"level": "M3", "title": "Director", "range": (160000, 250000)},
                        {"level": "M4", "title": "Senior Director", "range": (230000, 350000)},
                        {"level": "M5", "title": "VP", "range": (300000, 600000)},
                        {"level": "M6", "title": "SVP", "range": (500000, 900000)}
                    ]
                },
                "job_families": {
                    "engineering": {
                        "multiplier": 1.1,
                        "roles": ["Software Engineer", "DevOps Engineer", "Security Engineer"],
                        "skills_premium": {
                            "AI/ML": 0.12,
                            "Cloud Architecture": 0.10,
                            "Security": 0.08
                        }
                    },
                    "data_science": {
                        "multiplier": 1.12,
                        "roles": ["Data Scientist", "ML Engineer", "Data Analyst"],
                        "skills_premium": {
                            "Deep Learning": 0.12,
                            "NLP": 0.10,
                            "Computer Vision": 0.10
                        }
                    }
                },
                "leave_policy": {
                    "annual": {
                        "days": 25,
                        "accrual_rate": "2.08 days per month",
                        "carryover_limit": 5
                    },
                    "sick": {
                        "days": "statutory",
                        "paid": True,
                        "statutory_sick_pay": "£99.35 per week"
                    },
                    "parental": {
                        "primary": {
                            "weeks": 52,
                            "pay": "90% for 6 weeks, statutory for 33 weeks"
                        },
                        "secondary": {
                            "weeks": 2,
                            "pay": "100%"
                        }
                    }
                },
                "benefits": {
                    "health_insurance": {
                        "provider": "BUPA",
                        "coverage": "private",
                        "includes_dental": False,
                        "dental_option": "Additional cost"
                    },
                    "pension": {
                        "contribution": "5% employer",
                        "employee_minimum": "3%"
                    }
                }
            },
            "india": {
                "currency": "INR",
                "bands": {
                    "individual_contributor": [
                        {"level": "IC1", "title": "Associate", "range": (500000, 800000)},
                        {"level": "IC2", "title": "Professional", "range": (700000, 1200000)},
                        {"level": "IC3", "title": "Senior", "range": (1100000, 2000000)},
                        {"level": "IC4", "title": "Principal", "range": (1800000, 3500000)},
                        {"level": "IC5", "title": "Distinguished", "range": (3000000, 6000000)},
                        {"level": "IC6", "title": "Fellow", "range": (5000000, 9000000)}
                    ],
                    "management": [
                        {"level": "M1", "title": "Manager", "range": (1500000, 2500000)},
                        {"level": "M2", "title": "Senior Manager", "range": (2000000, 3500000)},
                        {"level": "M3", "title": "Director", "range": (3000000, 5000000)},
                        {"level": "M4", "title": "Senior Director", "range": (4500000, 7500000)},
                        {"level": "M5", "title": "VP", "range": (7000000, 15000000)},
                        {"level": "M6", "title": "SVP", "range": (12000000, 25000000)}
                    ]
                },
                "job_families": {
                    "engineering": {
                        "multiplier": 1.15,
                        "roles": ["Software Engineer", "DevOps Engineer", "Security Engineer"],
                        "skills_premium": {
                            "AI/ML": 0.18,
                            "Cloud Architecture": 0.15,
                            "Security": 0.12
                        }
                    },
                    "data_science": {
                        "multiplier": 1.2,
                        "roles": ["Data Scientist", "ML Engineer", "Data Analyst"],
                        "skills_premium": {
                            "Deep Learning": 0.18,
                            "NLP": 0.15,
                            "Computer Vision": 0.15
                        }
                    }
                },
                "leave_policy": {
                    "annual": {
                        "days": 24,
                        "accrual_rate": "2 days per month",
                        "carryover_limit": 10
                    },
                    "sick": {
                        "days": 12,
                        "paid": True
                    },
                    "parental": {
                        "primary": {
                            "weeks": 26,
                            "pay": "100%"
                        },
                        "secondary": {
                            "weeks": 2,
                            "pay": "100%"
                        }
                    }
                },
                "benefits": {
                    "health_insurance": {
                        "coverage": "family",
                        "includes_dental": True,
                        "sum_insured": "5 lakhs"
                    },
                    "provident_fund": {
                        "employer_contribution": "12%",
                        "employee_contribution": "12%"
                    }
                }
            }
        }

        # User groups with enhanced access controls
        self.user_groups = {
            "all_employees": {
                "description": "Basic access for all employees",
                "regions": ["global", "us", "uk", "india"],
                "access_levels": ["public"]
            },
            "hr_group": {
                "description": "HR team members",
                "regions": ["global", "us", "uk", "india"],
                "access_levels": ["public", "internal", "confidential"]
            },
            "finance_group": {
                "description": "Finance team members",
                "regions": ["global", "us", "uk", "india"],
                "access_levels": ["public", "internal", "confidential"]
            },
            "management": {
                "description": "Senior management and executives",
                "regions": ["global", "us", "uk", "india"],
                "access_levels": ["public", "internal", "confidential", "restricted"]
            },
            "hr_operations": {
                "description": "HR Operations team",
                "regions": ["global", "us", "uk", "india"],
                "access_levels": ["confidential", "restricted"]
            },
            "compensation_team": {
                "description": "Compensation and Benefits team",
                "regions": ["global", "us", "uk", "india"],
                "access_levels": ["confidential", "restricted", "classified"]
            }
        }

        # Document types configuration
        self.document_types = {
            "hr": self._generate_hr_templates(),
            "finance": self._generate_finance_templates(),
            "operations": self._generate_operations_templates(),
            "compensation": self._generate_compensation_templates()
        }

    def _load_env_vars(self):
        """Load and validate environment variables"""
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

    def _generate_hr_templates(self) -> Dict[str, Any]:
        """Generate HR document templates with regional variations"""
        base_path = "documents/hr"
        templates = self._create_base_template_structure(base_path)

        for region in self.regional_specs.keys():
            regional_specs = self.regional_specs[region]
            
            # Leave policy documents
            templates["public_content"].extend([
                {
                    "name": f"leave_policy_{region}.txt",
                    "groups": ["all_employees"],
                    "region": region,
                    "system_prompt": f"Creating leave policy documentation for {region.upper()}.",
                    "user_prompt": f"""Create comprehensive leave policy for {region.upper()}:
                    1. Leave Entitlements: {json.dumps(regional_specs['leave_policy'], indent=2)}
                    2. Leave Types and Eligibility
                    3. Leave Application Process
                    4. Holiday Calendar
                    5. Special Leave Categories
                    6. Leave Accrual and Carryover Rules"""
                }
            ])

            # Band structure documents
            templates["confidential_content"].extend([
                {
                    "name": f"band_structure_{region}.txt",
                    "groups": ["hr_group", "management", "compensation_team"],
                    "region": region,
                    "system_prompt": f"Creating band structure documentation for {region.upper()}.",
                    "user_prompt": f"""Create detailed band structure document for {region.upper()}:
                    1. Individual Contributor Bands: {json.dumps(regional_specs['bands']['individual_contributor'], indent=2)}
                    2. Management Bands: {json.dumps(regional_specs['bands']['management'], indent=2)}
                    3. Job Families and Multipliers: {json.dumps(regional_specs['job_families'], indent=2)}
                    4. Skills Premium Structure
                    5. Career Progression Framework
                    6. Band Transition Guidelines"""
                }
            ])

            # Benefits documents
            templates["internal_content"].extend([
                {
                    "name": f"benefits_guide_{region}.txt",
                    "groups": ["all_employees", "hr_group"],
                    "region": region,
                    "system_prompt": f"Creating benefits guide for {region.upper()}.",
                    "user_prompt": f"""Create comprehensive benefits guide for {region.upper()}:
                    1. Benefits Package: {json.dumps(regional_specs['benefits'], indent=2)}
                    2. Eligibility Criteria
                    3. Enrollment Process
                    4. Claims Procedures
                    5. Additional Benefits and Perks"""
                }
            ])

        return templates

    def _generate_finance_templates(self) -> Dict[str, Any]:
        """Generate finance document templates with regional variations"""
        base_path = "documents/finance"
        templates = self._create_base_template_structure(base_path)

        for region in self.regional_specs.keys():
            regional_specs = self.regional_specs[region]
            
            # Compensation budget documents
            templates["classified_content"].extend([
                {
                    "name": f"compensation_budget_{region}.txt",
                    "groups": ["finance_group", "management", "compensation_team"],
                    "region": region,
                    "system_prompt": f"Creating compensation budget documentation for {region.upper()}.",
                    "user_prompt": f"""Create detailed compensation budget document for {region.upper()}:
                    1. Salary Bands Budget: {json.dumps(regional_specs['bands'], indent=2)}
                    2. Benefits Cost Structure: {json.dumps(regional_specs['benefits'], indent=2)}
                    3. Annual Budget Planning
                    4. Quarterly Budget Reviews
                    5. Cost Optimization Strategies"""
                }
            ])

            # Financial planning documents
            templates["restricted_content"].extend([
                {
                    "name": f"financial_planning_{region}.txt",
                    "groups": ["finance_group", "management"],
                    "region": region,
                    "system_prompt": f"Creating financial planning documentation for {region.upper()}.",
                    "user_prompt": f"""Create comprehensive financial planning document for {region.upper()}:
                    1. Annual Budget Allocation
                    2. Headcount Planning
                    3. Operational Expenses
                    4. Investment Strategy
                    5. Risk Management"""
                }
            ])

        return templates

    def _generate_compensation_templates(self) -> Dict[str, Any]:
        """Generate compensation document templates with regional variations"""
        base_path = "documents/compensation"
        templates = self._create_base_template_structure(base_path)

        for region in self.regional_specs.keys():
            regional_specs = self.regional_specs[region]
            
            # Compensation strategy documents
            templates["classified_content"].extend([
                {
                    "name": f"compensation_strategy_{region}.txt",
                    "groups": ["compensation_team", "management"],
                    "region": region,
                    "system_prompt": f"Creating compensation strategy for {region.upper()}.",
                    "user_prompt": f"""Create detailed compensation strategy for {region.upper()}:
                    1. Salary Bands Structure: {json.dumps(regional_specs['bands'], indent=2)}
                    2. Job Family Framework: {json.dumps(regional_specs['job_families'], indent=2)}
                    3. Market Competitiveness Analysis
                    4. Pay Equity Guidelines
                    5. Performance-Based Compensation
                    6. Long-term Incentive Plans"""
                }
            ])

        return templates

    def _generate_operations_templates(self) -> Dict[str, Any]:
        """Generate operations document templates with regional variations"""
        base_path = "documents/operations"
        templates = self._create_base_template_structure(base_path)

        for region in self.regional_specs.keys():
            # Operational guidelines documents
            templates["internal_content"].extend([
                {
                    "name": f"operational_guidelines_{region}.txt",
                    "groups": ["all_employees"],
                    "region": region,
                    "system_prompt": f"Creating operational guidelines for {region.upper()}.",
                    "user_prompt": f"""Create comprehensive operational guidelines for {region.upper()}:
                    1. Office Policies
                    2. Work Hours and Flexibility
                    3. Equipment and Resources
                    4. Security Protocols
                    5. Emergency Procedures"""
                }
            ])

        return templates

    def _create_base_template_structure(self, base_path: str) -> Dict[str, Any]:
        """Create base template structure for document types"""
        return {
            "path": base_path,
            "public_content": [],
            "internal_content": [],
            "confidential_content": [],
            "restricted_content": [],
            "classified_content": []
        }

    async def generate_document_content(self, system_prompt: str, user_prompt: str) -> str:
        """Generate document content using Azure OpenAI"""
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
        """Create directory structure for documents"""
        for doc_type, info in self.document_types.items():
            Path(info["path"]).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {info['path']}")

    async def generate_all_documents(self) -> Dict[str, Any]:
        """Generate all documents with appropriate access controls"""
        self.create_directory_structure()
        metadata = {}
        
        for doc_type, info in self.document_types.items():
            logger.info(f"Generating {doc_type} documents...")
            
            for access_level in ["public_content", "internal_content", 
                               "confidential_content", "restricted_content", 
                               "classified_content"]:
                
                for template in info.get(access_level, []):
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
                            "access_level": access_level.replace("_content", ""),
                            "groups": template["groups"],
                            "region": template.get("region", "global"),
                            "created": datetime.utcnow().isoformat(),
                            "title": template["name"].replace(".txt", "").replace("_", " ").title(),
                            "last_modified": datetime.utcnow().isoformat()
                        }
                        
                        logger.info(f"Generated {access_level} document: {file_path}")
                        
                    except Exception as e:
                        logger.error(f"Error generating {file_path}: {str(e)}")
                        continue
        
        # Save metadata
        metadata_path = Path("documents/metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

async def main():
    """Main function to orchestrate document generation"""
    try:
        generator = DocumentGenerator()
        metadata = await generator.generate_all_documents()
        
        # Print detailed summary
        print("\nDocument Generation Summary")
        print("=" * 50)
        
        # Summary by region
        for region in ["us", "uk", "india"]:
            print(f"\n{region.upper()} Region Summary:")
            print("-" * 30)
            
            # Band structure summary
            print("\nBand Structure:")
            regional_specs = generator.regional_specs[region]
            
            print("\nIndividual Contributor Bands:")
            for band in regional_specs["bands"]["individual_contributor"]:
                print(f"{band['level']} - {band['title']}: {regional_specs['currency']} "
                      f"{band['range'][0]:,} - {band['range'][1]:,}")
            
            print("\nManagement Bands:")
            for band in regional_specs["bands"]["management"]:
                print(f"{band['level']} - {band['title']}: {regional_specs['currency']} "
                      f"{band['range'][0]:,} - {band['range'][1]:,}")
            
            # Job families summary
            print("\nJob Families:")
            for family, details in regional_specs["job_families"].items():
                print(f"\n{family.title()}:")
                print(f"Base Multiplier: {details['multiplier']}")
                if "skills_premium" in details:
                    print("Skills Premium:")
                    for skill, premium in details["skills_premium"].items():
                        print(f"  - {skill}: +{premium*100}%")
            
            # Leave policy summary
            print("\nLeave Policy:")
            print(json.dumps(regional_specs["leave_policy"], indent=2))
            
            # Benefits summary
            print("\nBenefits:")
            print(json.dumps(regional_specs["benefits"], indent=2))
            
            # Document count by access level
            print("\nGenerated Documents:")
            access_levels = ["public", "internal", "confidential", "restricted", "classified"]
            for level in access_levels:
                count = sum(1 for m in metadata.values() 
                          if m["region"] == region and m["access_level"] == level)
                print(f"{level.title()}: {count} documents")
        
        print("\nMetadata saved to: documents/metadata.json")
        
    except Exception as e:
        logger.error(f"Document generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())