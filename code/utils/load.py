# 1. Python standard libraries
import json

# 2. Other libraries
import torch
from transformers import CLIPProcessor, CLIPModel
import lpips

from diffusers import TextToVideoZeroPipeline


def load_t2vz(device="cuda"):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe_t2vz = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return pipe_t2vz


def load_clip(device="cuda"):
    # Score Model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

def load_lpips():
    lpips_model = lpips.LPIPS(net="vgg")
    return lpips_model

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data
