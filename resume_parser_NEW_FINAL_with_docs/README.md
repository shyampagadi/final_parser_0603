# Resume Parser & Matching System

A powerful resume parsing and job description matching system using AWS Bedrock embedding models and semantic search.

## Overview

This system helps recruiters and hiring managers efficiently match job descriptions with the most relevant candidate resumes using advanced vector embeddings and natural language understanding.

### Key Features

- **Resume Parsing**: Extract structured data from resumes in various formats (PDF, DOCX, TXT)
- **Semantic Search**: Find candidates based on the meaning of job requirements, not just keywords
- **Multi-factor Ranking**: Rank candidates based on skills, experience, and semantic relevance
- **AWS Integration**: Leverages AWS Bedrock, OpenSearch, S3, and optionally DynamoDB
- **Customizable Matching**: Adjust weights and criteria to find the best candidates for each role

## Getting Started

### Prerequisites

- Python 3.9+
- AWS account with Bedrock and OpenSearch access
- Proper AWS credentials configuration

### Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   cp .env-example .env
   # Edit .env with your AWS credentials and configuration
   ```

3. Parse resumes:
   ```bash
   python parse_resume.py
   ```

4. Match with job description:
   ```bash
   python retrieve_jd_matches.py --jd_file job_descriptions/software_engineer.txt
   ```

For detailed setup and usage instructions, see the [Documentation](./docs/README.md).

## System Architecture

The system uses a multi-tiered architecture:

- **AWS Bedrock**: Generates vector embeddings for semantic search
- **Amazon OpenSearch**: Stores and searches vector embeddings
- **Amazon S3**: Stores raw and processed resume files
- **Optional Databases**:
  - **PostgreSQL**: Stores PII and detailed candidate information
  - **DynamoDB**: Stores structured resume data

## Documentation

Comprehensive documentation is available in the [docs](./docs/README.md) directory:

- [Getting Started Guide](./docs/user-guides/getting-started.md)
- [System Architecture](./docs/architecture/system-architecture.md)
- [Vector Embedding Framework](./docs/technical-docs/vector-embedding-framework.md)
- [Reranking Algorithm](./docs/technical-docs/reranking-algorithm.md)
- [Troubleshooting Guide](./docs/troubleshooting/common-issues.md)

## Example Usage

```python
# Initialize the resume retriever
retriever = ResumeRetriever()

# Search for candidates matching a job description
results = retriever.search_resumes(
    jd_file="job_descriptions/senior_developer.txt",
    method="vector",
    size=20
)

# Process and display results
for candidate in results["matches"]:
    print(f"Candidate: {candidate['name']} - Score: {candidate['score']}")
    print(f"  Matched Skills: {', '.join(candidate['matched_skills'])}")
    print(f"  Experience: {candidate['years_of_experience']} years")
```

## Tools

- **parse_resume.py**: Process and parse resumes from various formats
- **retrieve_jd_matches.py**: Match job descriptions with parsed resumes
- **scripts/create_job_description.py**: Helper to create well-formatted job descriptions

## Command Line Reference

### Resume Parsing

```bash
python parse_resume.py [--file FILE] [--reindex] [--force] [--verbose]
```

### Job Description Matching

```bash
python retrieve_jd_matches.py --jd_file JD_FILE [--method {vector,text,hybrid}] 
                              [--max MAX] [--exp EXP] [--no-rerank]
                              [--weights WEIGHTS] [--output OUTPUT]
```

## License

MIT License

## Acknowledgements

- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [OpenSearch](https://opensearch.org/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Spacy](https://spacy.io/) 