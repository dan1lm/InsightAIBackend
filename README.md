# The backend for InsightAI. 

Uses FastAPI uses the Llama 2 model (7B parameters) from Hugging Face.

### API Endpoints
- `POST /generate-insight`: Generates AI insights from provided health data
- `GET /health`: Server health check and model status

### Setup Requirements
1. Hugging Face account and access token
2. Python 3.8+
3. GPU recommended for optimal performance
4. Environment variables:
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token

### Running the Server:

- uvicorn main:app --reload
