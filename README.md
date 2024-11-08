# FAISS Server for Embedding Management and Similarity Search

This project is a Python server that uses [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) to manage and perform similarity searches on embeddings. It leverages Flask to provide two main endpoints for adding embeddings and performing similarity searches.

## Features
- **Add Embeddings**: Upload embeddings and their corresponding chunks to the FAISS index.
- **Search Embeddings**: Perform similarity search on embeddings to retrieve the most relevant chunks.

## Requirements

- Python 3.6+
- FAISS library
- Flask

## Installation

1. **Clone the repository**:
```
   git clone https://github.com/yourusername/faiss-server.git
   cd faiss-server
```

2. ** Install dependencies: **
Install the required Python packages by running:

`pip install -r requirements.txt`

Note: Ensure that FAISS is installed. You can install it with:
`pip install faiss-cpu`

3. Run the server:

`python faiss_server.py`

4. **Access the API:** By default, the server runs on http://127.0.0.1:5000.

## API Endpoints
1. /add_embeddings [POST]
- Description: Adds embeddings to the FAISS index and stores associated chunks.
- Request Body:
- embeddings: A list of embedding vectors to be added to FAISS.
- chunks: Corresponding text chunks for each embedding.
- Response: JSON status message confirming success or failure.
2. /search_embeddings [POST]
- Description: Searches for the most similar embeddings in FAISS and retrieves relevant chunks.
- Request Body:
- embedding: A single embedding vector to query against the index.
- Response:
- results: A list of chunks with the most similarity to the query embedding.
  
## Usage Example
### Adding Embeddings
```
POST /add_embeddings
{
  "embeddings": [[0.1, 0.2, 0.3, ...], ...],
  "chunks": ["This is the first chunk of text.", "This is the second chunk.", ...]
}
```
### Searching Embeddings
```
POST /search_embeddings
{
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

### Data Persistence
- FAISS Index: Stored in faiss_index.index for persistence across server restarts.
- Chunks: Stored in chunk_store.json.
  
## License
- This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- FAISS by Facebook AI Research
- Flask for the lightweight server framework.
