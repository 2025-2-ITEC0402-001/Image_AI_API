import torch
from diffusers import FluxPipeline
import io
import logging
from flask import Flask, request, send_file, jsonify

# --- 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipe = None

# --- 모델 로딩 ---
def load_models():
    global pipe
    if pipe is None:
        logger.info("Loading FluxPipeline for generation...")
        lora_path = "/abr/coss33/.cache/huggingface/hub/models--ali-vilab--In-Context-LoRA/snapshots/16dae427a8509229309b85bc5345dfeffee5fc2e/film-storyboard.safetensors"
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()
        logger.info("FluxPipeline (Generation) loaded successfully.")

# 앱이 시작되고 첫 요청이 들어올 때 모델을 로드합니다.
@app.before_request
def before_request():
    load_models()

# --- API 엔드포인트 ---
@app.route("/generate-storyboard", methods=["POST"])
def generate_storyboard():
    # 요청 데이터 가져오기
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is missing"}), 400

    prompt = data.get('prompt')
    height = data.get('height', 1536)
    width = data.get('width', 1024)
    guidance_scale = data.get('guidance_scale', 3.5)
    num_inference_steps = data.get('num_inference_steps', 20)
    seed = data.get('seed', 0)

    # 이미지 생성
    logger.info("Generating image...")
    generator = torch.Generator("cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator
    ).images[0]

    # 이미지를 메모리 버퍼에 저장하여 반환
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')
