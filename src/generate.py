from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

init_image = Image.open("init_image.png").convert("RGB").resize((512, 512))
mask_image = Image.open("mask_image.png").convert("RGB").resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16
).to("cuda")

output = pipe(
    prompt="restore eyes, same person, open eyes, photorealistic",
    image=init_image,
    mask_image=mask_image,
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]

output.save("output_fixed.png")
