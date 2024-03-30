import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from utils import generate_image, load_pipeline

ip_model = load_pipeline()
executor = ThreadPoolExecutor()
app = FastAPI()


def create_image(image_content, prompt, negative_prompt):
    """
    Process an image using the provided content, prompt, and negative prompt.

    Args:
        image_content (str): The content of the image to be processed.
        prompt (str): The prompt for processing the image.
        negative_prompt (str): The negative prompt for processing the image.

    Returns:
        processed_image: The processed image.

    Raises:
        HTTPException: If an HTTPException occurs.
    """
    try:
        processed_image = generate_image(
            image_content, prompt, negative_prompt, ip_model
        )
        return processed_image
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # Handle exceptions, log errors, and return an appropriate response
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def create_image_async(image_content, prompt, negative_prompt):
    """
    Process an image asynchronously.

    Args:
        image_content (bytes): The content of the image to be generate image.
        prompt (str): The prompt to use for image creation.
        negative_prompt (str): Negative prompt to be used to create an image.

    Returns:
        The generated image.

    Raises:
        None.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, create_image, image_content, prompt, negative_prompt
    )


@app.get("/")
async def read_root():
    """
    Endpoint for root path, returns a health check response.

    Returns:
        dict: A dictionary indicating the health check status.
    """
    return {"health_check": "OK"}


@app.post("/generate_image")
async def generate(image: UploadFile, prompt: str, negative_prompt: str):
    """
    Generate an image based on the provided prompt and negative prompt.

    Parameters:
        image (UploadFile): The image file to generate from.
        prompt (str): The prompt to use for generating the image.
        negative_prompt (str): The negative prompt to use for generating the image.

    Returns:
        FileResponse: The generated image as a FileResponse object.
    """
    generated_img_path = await create_image_async(
        image.file.read(), prompt, negative_prompt
    )
    return FileResponse(generated_img_path, media_type="image/png")
