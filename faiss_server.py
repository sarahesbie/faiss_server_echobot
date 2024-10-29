from flask import Flask, request, jsonify
import faiss
import numpy as np

app = Flask(__name__)

# Initialize FAISS index and chunk storage
index = faiss.IndexFlatL2(1536)  # Dimension size for embeddings
chunk_store = []  # List to store chunks corresponding to embeddings

# Endpoint to add embeddings to the FAISS index
@app.route('/add_embeddings', methods=['POST'])
def add_embeddings():
    try:
        print("Received request at /add_embeddings")

        # Parse the embeddings and chunks from the request data
        data = request.get_json()
        if not data:
            print("Error: No data received in the request.")
            raise ValueError("No data received in the request.")

        embeddings = data.get('embeddings')
        chunks = data.get('chunks')

        if embeddings is None or chunks is None:
            print("Error: Embeddings or chunks missing in request data.")
            raise ValueError("Embeddings or chunks missing in request data.")

        print(f"Number of embeddings received: {len(embeddings)}")
        print(f"Number of chunks received: {len(chunks)}")

        # Convert embeddings to a NumPy array
        embeddings_array = np.array(embeddings, dtype='float32')
        print("Embeddings array shape:", embeddings_array.shape)

        # Add the embeddings to the FAISS index
        index.add(embeddings_array)
        print("Embeddings successfully added to FAISS index.")

        # Store the chunks in chunk_store
        chunk_store.extend(chunks)
        print("Chunks successfully stored.")

        # Return a success response
        return jsonify({"status": "success", "message": "Embeddings added to FAISS"}), 200

    except Exception as e:
        print("Error in /add_embeddings:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# Endpoint to search embeddings in the FAISS index
@app.route('/search_embeddings', methods=['POST'])
def search_embeddings():
    try:
        print("Received request at /search_embeddings")

        # Retrieve the embedding vector from the request data
        data = request.get_json()
        if not data:
            print("Error: No data received in the request.")
            raise ValueError("No data received in the request.")

        query_embedding = data.get('embedding')
        if query_embedding is None:
            print("Error: Embedding missing in request data.")
            raise ValueError("Embedding missing in request data.")

        query_vector = np.array([query_embedding], dtype='float32')
        print("Query vector shape:", query_vector.shape)

        # Perform the search in FAISS
        k = 15  # Number of nearest neighbors to retrieve; adjust as needed
        distances, indices = index.search(query_vector, k)
        print(f"Search results: distances={distances}, indices={indices}")

        # Retrieve the corresponding chunks from `chunk_store`
        results = []
        for idx in indices[0]:
            if idx != -1:  # Check if index is valid
                chunk_text = chunk_store[idx]  # Retrieve the chunk from storage
                results.append(chunk_text)

        # Return the retrieved chunks
        print("Search results retrieved successfully.")
        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print("Error in /search_embeddings:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
