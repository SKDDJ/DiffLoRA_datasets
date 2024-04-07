from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from pathlib import Path
import re

def slugify(text):
    # remove non-word characters and foreign characters
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text

model_id = "stabilityai/stable-diffusion-2"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to(device)

DIR_NAME="./images/"
dirpath = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath.mkdir(parents=True, exist_ok=True)

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "blurry, dark photo, blue"
steps = 15
scale = 9
num_images_per_prompt = 5
seed = torch.randint(0, 1000000, (1,)).item()
generator = torch.Generator(device=device).manual_seed(seed)
output = pipe(prompt, negative_prompt=negative_prompt, width=512, height=512, num_inference_steps=steps,
             guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)

for idx, image in enumerate(output.images):
    image_name = f'{slugify(prompt)}-{idx}.png'
    image_path = dirpath / image_name
    image.save(image_path)