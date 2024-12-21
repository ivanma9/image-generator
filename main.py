import modal
from datetime import datetime, timezone
import os
import io
from fastapi import Response, Query, Request, HTTPException
import requests
def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch


    AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")



image = (modal.Image.debian_slim().pip_install("diffusers", "fastapi[standard]", "transformers", "accelerate", "requests").run_function(download_model))


app = modal.App("sd-demo", image=image)

@app.cls(image=image, gpu="A10G", container_idle_timeout=600, secrets=[modal.Secret.from_name("API_KEY")])
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
        self.API_KEY = os.environ["CLIENT_TOKEN_2"]
    
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt to generate an image from")):
        print("prompt", prompt)
        print("headers", request.headers)

        api_key = request.headers.get("X-API-Key")
        if api_key != self.API_KEY:
            print(self.API_KEY)
            print("api_key", api_key)
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        print(f"Generate success: {datetime.now(timezone.utc).isoformat()}")
        return Response(buffer.getvalue(), media_type="image/jpeg")

    @modal.web_endpoint()
    def health(self):
        """Lightweight healthcheck to keep the container warm"""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.function(schedule=modal.Cron("*/10 * * * *"), secrets=[modal.Secret.from_name("API_KEY")])
def keep_warm():
    health_url = "https://ivanma9--sd-demo-model-health.modal.run"
    generate_url = "https://ivanma9--sd-demo-model-generate.modal.run"
    """Keeps the model container warm by running an inference every 30 minutes"""
    health_response = requests.get(health_url)
    print(f"health_response: {health_response.json()['timestamp']}")
    
    headers = {"X-API-Key": os.environ["CLIENT_TOKEN_2"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generate endpoint success: {datetime.now(timezone.utc).isoformat()}")
    



