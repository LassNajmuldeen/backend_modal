import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

# Download model from huggingface
def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
    )
    
# Create docker container that uses linux
image = (modal.Image.debian_slim()
        .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests")
        .run_function(download_model))

app = modal.App("sd-turbo", image=image)

@app.cls(
    image=image,
    gpu="A10G",
)
class Model:

    @modal.build() # Call when modal app is building
    @modal.enter() # Call when container starts
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        # Define again
        self.pipe = AutoPipelineForText2Image(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )

        self.pipe.to("cuda")