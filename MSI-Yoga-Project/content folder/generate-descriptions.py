import json
import time
import logging
import vertexai
from langchain_google_vertexai import VertexAI
from tenacity import retry, stop_after_attempt, wait_exponential
from settings import get_settings

settings = get_settings()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Initialize Vertex AI SDK
vertexai.init(project=settings.project_id, location=settings.location)
logging.info("Done Initializing Vertex AI SDK")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def generate_description(pose_name, sanskrit_name, expertise_level, pose_types):
    """Generates a description for a yoga pose using the Gemini API."""

    prompt = f"""
    Generate a concise description (max 50 words) for the yoga pose: {pose_name}
    Also known as: {sanskrit_name}
    Expertise Level: {expertise_level}
    Pose Type: {", ".join(pose_types)}

    Include key benefits and any important alignment cues.
    """
    try:
        model = VertexAI(model_name=settings.gemini_model_name, verbose=True)
        response = model.invoke(prompt)
        return response
    except Exception as e:
        logging.info(f"Error generating description for {pose_name}: {e}")
        return ""


def add_descriptions_to_json(input_file, output_file):
    """Loads JSON data, adds descriptions, and saves the updated data."""

    with open(input_file, "r") as f:
        yoga_poses = json.load(f)

    total_poses = len(yoga_poses)
    processed_count = 0

    for pose in yoga_poses:
        if pose["name"] != " Pose":
            start_time = time.time()  # Record start time
            pose["description"] = generate_description(
                pose["name"],
                pose["sanskrit_name"],
                pose["expertise_level"],
                pose["pose_type"],
            )
            end_time = time.time()  # Record end time

            processed_count += 1
            end_time = time.time()  # Record end time
            time_taken = end_time - start_time
            logging.info(
                f"Processed: {processed_count}/{total_poses} - {pose['name']} ({time_taken:.2f} seconds)"
            )

        else:
            pose["description"] = ""
            processed_count += 1
            logging.info(
                f"Processed: {processed_count}/{total_poses} - {pose['name']} ({time_taken:.2f} seconds)"
            )
        # Adding a delay to avoid rate limit
        time.sleep(30)

    with open(output_file, "w") as f:
        json.dump(yoga_poses, f, indent=2)


def main():
    # File paths
    input_file = "./data/yoga_poses.json"
    output_file = "./data/yoga_poses_with_descriptions.json"

    # Add descriptions and save the updated JSON
    add_descriptions_to_json(input_file, output_file)


if __name__ == "__main__":
    main()
