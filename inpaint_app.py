import torch
from diffusers import FluxInpaintPipeline
from PIL import Image, ImageDraw
import io, logging, uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

app = FastAPI(title="FLUX Inpaint API")
pipe = None

@app.on_event("startup")
def load_models():
    global pipe
    logger.info("Loading FluxInpaintPipeline for revision...")
    lora_path = "/abr/coss33/.cache/huggingface/hub/models--ali-vilab--In-Context-LoRA/snapshots/16dae427a8509229309b85bc5345dfeffee5fc2e/film-storyboard.safetensors"
    pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.enable_model_cpu_offload()
    logger.info("FluxInpaintPipeline (Revision) loaded successfully.")

@app.post("/revise-storyboard", response_class=StreamingResponse)
async def revise_storyboard(image: UploadFile = File(...), revised_prompt: str = Form(...), strength: float = Form(0.9), guidance_scale: float = Form(3.5), num_inference_steps: int = Form(25), seed: int = Form(0)):
    logger.info("Revising image...")
    original_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    width, height = original_image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([(0, height / 3), (width, height * 2 / 3)], fill=255)
    generator = torch.Generator("cpu").manual_seed(seed)
    final_image = pipe(prompt=revised_prompt, image=original_image, mask_image=mask, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator, width=width, height=height).images[0]
    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")
