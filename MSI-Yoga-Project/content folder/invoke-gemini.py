import logging
from langchain_google_vertexai import VertexAI
import vertexai
from settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    try:
        # Initialize Vertex AI SDK
        logging.info("Initializing Vertex AI SDK")
        vertexai.init(project=settings.project_id, location=settings.location)
        logging.info("Done Initializing Vertex AI SDK")
        model = VertexAI(model_name=settings.gemini_model_name, verbose=True)
        response = model.invoke("Tell me something about Yoga")
        return response
    except Exception as e:
        logging.error(f"Error invoking Gemini: {e}")
        return None


if __name__ == "__main__":
    print(main())
