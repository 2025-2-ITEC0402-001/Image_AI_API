import torch
from diffusers import FluxPipeline
import io, logging, uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # <-- 이 줄이 추가되었습니다.

app = FastAPI(title="FLUX Generate API")
pipe = None

@app.on_event("startup")
def load_models():
    global pipe
    logger.info("Loading FluxPipeline for generation...")
    lora_path = "/abr/coss33/.cache/huggingface/hub/models--ali-vilab--In-Context-LoRA/snapshots/16dae427a8509229309b85bc5345dfeffee5fc2e/film-storyboard.safetensors"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.enable_model_cpu_offload()
    logger.info("FluxPipeline (Generation) loaded successfully.")

class StoryboardRequest(BaseModel):
    prompt: str
    height: int = 1536
    width: int = 1024
    guidance_scale: float = 3.5
    num_inference_steps: int = 20
    seed: int = 0

@app.post("/generate-storyboard", response_class=StreamingResponse)
async def generate_storyboard(request: StoryboardRequest):
    logger.info("Generating image...")
    generator = torch.Generator("cpu").manual_seed(request.seed)
    image = pipe(prompt=request.prompt, height=request.height, width=request.width, guidance_scale=request.guidance_scale, num_inference_steps=request.num_inference_steps, max_sequence_length=512, generator=generator).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")
