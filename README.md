# OCR_RAG_CHATBOT
This chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate and context-aware answers by retrieving and generating responses based on text extracted via OCR from documents.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG_OCR_Assistant
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3.  Download and start the Llama model using Ollama:
```bash
ollama pull llama3
ollama run llama3
```

## Project Structure

  Ensure you have the following files in place:

- **merged_index.faiss:** FAISS index file containing document embeddings  
- **merged_metadata.pkl:** Metadata file containing document content  
- **index.html:** Web interface for the chatbot  
- **app.py:** Main Flask application  


## Running The Application

   1.Make sure Ollama is running with the llama3 model

   2.Navigate to the Chatbot directory and start the Flask application:
   ```bash
   cd Chatbot
   python app.py
   ```

   3.Open your web browser and navigate to:
   ```bash
   http://localhost:5000
   ```





