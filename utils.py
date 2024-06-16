import io
import os

import numpy as np
import requests
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import make_image_grid
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from PIL import Image
from tqdm import tqdm


def save_images(images):
    """
    Saves a list of images by creating an image grid and saving it as 'output/image_grid.jpg'.

    Parameters:
    images (list): A list of PIL image objects to be saved.

    Returns:
    None
    """
    make_image_grid(images, 2, 2).save("output/image_grid.jpg")
    for idx, image in enumerate(tqdm(images)):
        image.save(f"output/image_{idx}.jpg")


def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    :param directory_path: The path of the directory to be created.
    :return: None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_face_embedding(uploaded_img):
    """
    Create a face embedding for a given uploaded image.

    Parameters:
        uploaded_img (bytes): The uploaded image in bytes format.

    Returns:
        faceid_embeds (torch.Tensor): The face embedding of the image.
        face_image (numpy.ndarray): The normalized and cropped face image.
    """
    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    image = Image.open(io.BytesIO(uploaded_img))
    R, G, B = image.split()
    image_array = np.array(Image.merge("RGB", (B, G, R)))
    faces = app.get(image_array)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(
        image_array, landmark=faces[0].kps, image_size=224
    )
    return faceid_embeds, face_image


def load_pipeline():
    """
    Loads the pipeline for the image processing model.

    Returns:
        ip_model (IPAdapterFaceIDPlus): The loaded image processing model.
    """
    base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    ip_ckpt = "embeddings/ip-adapter-faceid-plus_sd15.bin"
    device = "cuda"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    realisticvision_negative_embeds = "embeddings/realisticvision-negative-embedding.pt"
    bad_dream_negative_embeds = "embeddings/BadDream.pt"
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    pipe.load_textual_inversion(
        realisticvision_negative_embeds, token="realisticvision-negative-embedding"
    )
    pipe.load_textual_inversion(bad_dream_negative_embeds, token="BadDream")
    ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)
    return ip_model


def generate_image(uploaded_img, prompt, negative_prompt, ip_model):
    """
    Generate an image based on the uploaded image, a prompt, a negative prompt, and an image processing model.

    Parameters:
    - uploaded_img: the image to generate the new image from
    - prompt: the main prompt for image generation
    - negative_prompt: additional prompt for negative aspects
    - ip_model: the image processing model to use for generation

    Returns:
    - The file path of the generated image
    """
    faceid_embeds, face_image = create_face_embedding(uploaded_img)

    prompt_with_template = f"RAW photo,{prompt},natural skin, 8k uhd, high quality, film grain, Fujifilm XT3,wide angle shot"
    negative_prompt = f"{negative_prompt},(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,realisticvision-negative-embedding,BadDream,NSFW:1.4"

    images = ip_model.generate(
        prompt=prompt_with_template,
        negative_prompt=negative_prompt,
        face_image=face_image,
        faceid_embeds=faceid_embeds,
        shortcut=False,
        s_scale=2.0,
        num_samples=1,
        width=512,
        height=768,
        num_inference_steps=30,
    )
    create_directory("output")
    images[0].save("output/image_to_show.png")
    return "output/image_to_show.png"


def get_image(prompt, negative_prompt, image_file):
    url = "http://127.0.0.1:8000/generate_image"
    files = {"image": ("image.png", image_file, "image/png")}
    params = {"prompt": prompt, "negative_prompt": negative_prompt}

    response = requests.post(url, files=files, params=params)
    if response.status_code == 200:
        return response.content
