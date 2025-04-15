import logging
from flask import Flask, request, jsonify, render_template, make_response
from google.cloud import firestore
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import google.cloud.texttospeech as tts
import urllib
from settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

def search(query: str):
    """Executes Firestore Vector Similarity Search"""
    embedding = VertexAIEmbeddings(
        model_name=settings.embedding_model_name,
        project=settings.project_id,
        location=settings.location,
    )

    client = firestore.Client(
        project=settings.project_id, database=settings.database
    )

    vector_store = FirestoreVectorStore(
        client=client,
        collection=settings.collection,
        embedding_service=embedding,
    )

    logging.info(f"Now executing query: {query}")
    results: list[Document] = vector_store.similarity_search(
        query=query, k=int(settings.top_k), include_metadata=False
    )
    
    #Format the results for JSON output
    formatted_results = [
        {"page_content": result.page_content, "metadata": result.metadata} for result in results
    ]

    return formatted_results

@app.route("/", methods=["GET"])
def index():
    """Renders the HTML page with the search form."""
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_api():
    """API endpoint to search documents based on the prompt provided in json"""
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing prompt in request body"}), 400
        prompt = data["prompt"]
        results = search(prompt)
        return jsonify({"results": results}), 200
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return jsonify({"error": "An error occurred during search"}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    data = request.get_json()
    description = data.get('description')
    if not description:
        return jsonify({'error': 'Missing description'}), 400

    try:
        # Modify this to match your voice settings
        decoded_description = urllib.parse.unquote(description)
        voice_name = "en-US-Wavenet-D"
        audio_content = text_to_wav(voice_name, decoded_description)  #Call your TTS function

        #return audio_content directly as a response
        response = make_response(audio_content)
        response.headers.set('Content-Type', 'audio/wav')
        response.headers.set('Content-Disposition', 'attachment', filename='audio.wav')

        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    return response.audio_content

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(settings.port))
