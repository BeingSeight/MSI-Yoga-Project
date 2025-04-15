import logging
from settings import get_settings
import vertexai
from vertexai.vision_models import ImageGenerationModel

settings = get_settings()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

vertexai.init(project=settings.project_id, location=settings.location)


def generate_image(prompt: str):
    try:
        model = ImageGenerationModel.from_pretrained(
            settings.image_generation_model_name
        )

        images_Response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
        )

        if len(images_Response.images) == 0:
            logging.error("No images generated")
            return None

        # generate a random output file name with an extension being png
        import uuid

        output_file = f"./images/{uuid.uuid4().hex}.png"

        images_Response.images[0].save(
            location=output_file, include_generation_parameters=False
        )

        # Optional. View the generated image in a notebook.
        # images[0].show()

        print(
            f"Created output image using {len(images_Response.images[0]._image_bytes)} bytes"
        )
        # Example response:
        # Created output image using 1234567 bytes
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None


if __name__ == "__main__":
    generate_image(prompt="Generate photo of Indian flag being unfurled")
