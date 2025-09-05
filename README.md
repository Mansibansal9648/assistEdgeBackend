# assistEdgeBackend

## Overview
assistEdgeBackend is an open-source AI-powered knowledge bot built using FastAPI. It allows users to upload documents, extract text, generate embeddings, and query the knowledge base for relevant information. The backend leverages FAISS for similarity search and Hugging Face models for text generation.

## Features
- Upload documents in various formats (TXT, DOCX, PPTX, PDF).
- Extract and chunk text from uploaded documents.
- Generate embeddings using Sentence Transformers.
- Store and query knowledge base using FAISS.
- Generate answers to user queries using Hugging Face models.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- pip

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mansibansal9648/assistEdgeBackend.git
   cd assistEdgeBackend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

3. Use the following endpoints:
   - `POST /upload/`: Upload a document to the knowledge base.
   - `POST /query/`: Query the knowledge base for relevant information.
   - `GET /download/{doc_id}`: Download a specific document by its ID.

## Example
### Upload a Document
Use the `/upload/` endpoint to upload a document. Supported formats include TXT, DOCX, PPTX, and PDF.

### Query the Knowledge Base
Send a query to the `/query/` endpoint to retrieve relevant information from the uploaded documents.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments
- [FastAPI](https://fastapi.tiangolo.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/transformers/)