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

    pipe = AutoPipelineForText2Image.from_pretrained(
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
    secrets=[modal.Secret.from_name("API_KEY")]
)
class Model:

    @modal.build() # Call when modal app is building
    @modal.enter() # Call when container starts
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        # Define again
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )

        self.pipe.to("cuda")
        self.API_KEY = os.environ["API_KEY"]

    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):

        api_key = request.headers.get("X-API-Key")
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"

            )

        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        # Return the image or convert to a suitable response
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    
    @modal.web_endpoint()
    def health(self):
        """Endpoint for keeping the container warm"""
        return{"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    
@app.function(
        schedule=modal.Cron("*/5 * * * *"), # run every 5 mins
        secrets=[modal.Secret.from_name("API_KEY")]
)  
# function to prevent cold start
def update_keep_warm():
    health_url = "https://lassnajmuldeen--sd-turbo-model-health.modal.run"
    generate_url = "https://lassnajmuldeen--sd-turbo-model-generate.modal.run"
    
    # Check health endpoint first
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")

    # Create a test request to geenrate endpoint with API_KEY
    headers = {"X-API-Key": os.environ["API_KEY"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generate endpoint successfully tested at: {datetime.now(timezone.utc).isoformat()}")
