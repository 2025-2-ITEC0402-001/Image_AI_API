# 이 파일이 API 서버를 실행하고 요청을 받는 파일(3번 영상 생성)
import fastapi
import uvicorn
import shutil
import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import traceback

# demo.py의 로직을 import
try:
    from pipeline_logic import load_pipeline, run_inference_on_demand
except ImportError:
    print("="*50)
    print("오류: pipeline_logic.py를 찾을 수 없습니다.")
    print("fastapi_server.py와 pipeline_logic.py가 같은 폴더에 있는지 확인하세요.")
    print("="*50)
    exit(1)
except ModuleNotFoundError as e:
    print("="*50)
    print(f"오류: 모듈 임포트 실패 ({e})")
    print("Conda 환경 (motioni2v)이 활성화되었는지 확인하세요.")
    print("스크립트를 'DragAnything' 프로젝트 루트 폴더에서 실행 중인지 확인하세요.")
    print("="*50)
    exit(1)


app = FastAPI(title="Motion Generation API", version="1.0")

# --- 0. 설정 ---
# 비디오 생성을 위한 기본 설정 (demo.py의 args와 동일하게)
# (이 경로들은 fastapi_server.py를 실행하는 위치 기준의 상대 경로임)
PIPELINE_ARGS = {
    "pretrained_model_name_or_path": "stabilityai/stable-video-diffusion-img2vid",
    "DragAnything": "./model_out/DragAnything",
    "model_DIFT": "./utils/pretrained_models/chilloutmix",
    
    # API에서 조절 가능하게 만들 수도 있지만, 일단 고정값으로 설정
    "height": 320,
    "width": 576,
    "frame_number": 14 
}

# 작업 파일(입력/출력)을 저장할 기본 디렉터리
JOBS_DIR = "api_jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# 생성된 비디오를 다운로드할 수 있도록 /results URL을 JOBS_DIR에 매핑
app.mount("/results", StaticFiles(directory=JOBS_DIR), name="results")

# 작업 상태를 추적할 간단한 인메모리 DB
job_db = {}

# --- 1. 모델 로딩 (서버 시작 시 1회) ---
# 서버가 켜질 때 모델을 미리 로드합니다.
print("API 서버가 시작됩니다... 모델을 로드합니다 (시간이 걸릴 수 있습니다)...")
try:
    PIPELINE = load_pipeline(PIPELINE_ARGS)
except Exception as e:
    print("="*50)
    print(f"치명적 오류: 파이프라인 로딩 실패! {e}")
    print("모델 경로 (DragAnything, model_DIFT)가 올바른지 확인하세요.")
    print("="*50)
    PIPELINE = None # 로딩 실패 표시

@app.on_event("startup")
async def startup_event():
    if PIPELINE is None:
        print("모델이 로드되지 않았습니다. API가 정상 작동하지 않을 수 있습니다.")
    else:
        print("모델 로드 완료. API 서버가 준비되었습니다.")

# --- 2. 백그라운드 작업 함수 ---
def run_generation_task(
    job_id: str, 
    job_dir: str, 
    image_path: str, 
    mask_path: str, 
    traj_path: str,
    frame_number: int # API로 받은 frame_number
):
    """
    FastAPI의 BackgroundTasks가 이 함수를 백그라운드에서 실행합니다.
    """
    global job_db, PIPELINE, PIPELINE_ARGS
    
    try:
        if PIPELINE is None:
            raise RuntimeError("파이프라인이 로드되지 않았습니다.")
            
        print(f"[Job {job_id}] 백그라운드 작업 시작...")
        
        # API로 받은 frame_number를 사용하기 위해 args 복사 및 수정
        task_args = PIPELINE_ARGS.copy()
        task_args["frame_number"] = frame_number

        # 실제 추론 로직 호출
        output_video_path = run_inference_on_demand(
            pipeline=PIPELINE,
            pipeline_args=task_args,
            job_dir=job_dir,
            origin_image_path=image_path,
            mask_path=mask_path,
            trajectory_path=traj_path
        )
        
        # 결과물의 상대 경로 (다운로드 URL)
        # os.path.relpath는 플랫폼(윈도우/리눅스)에 따라 '\' 또는 '/'를 사용
        relative_video_url = os.path.relpath(output_video_path, JOBS_DIR)
        
        # DB에 '완료' 상태 및 다운로드 URL 저장 (URL은 항상 '/' 사용)
        job_db[job_id] = {
            "status": "complete",
            "video_url": f"/results/{relative_video_url.replace(os.path.sep, '/')}"
        }
        print(f"[Job {job_id}] 작업 완료.")

    except Exception as e:
        error_message = traceback.format_exc()
        print(f"[Job {job_id}] 작업 실패: {error_message}")
        job_db[job_id] = {
            "status": "error",
            "message": str(e) # 사용자에게 보여줄 간단한 에러 메시지
        }

# --- 3. API 엔드포인트 정의 ---

@app.post("/generate", summary="비디오 생성 요청 (비동기)")
async def start_generation(
    background_tasks: BackgroundTasks,
    origin_image: UploadFile = File(..., description="원본 이미지 (demo.jpg 역할)"),
    sam_mask: UploadFile = File(..., description="SAM 마스크 이미지 (mask_name 역할)"),
    trajectory_data: UploadFile = File(..., description="궤적 JSON 파일 ([[x1, y1], ...])"),
    frame_number: int = Form(PIPELINE_ARGS["frame_number"], description="생성할 총 프레임 수")
):
    """
    **입력 (multipart/form-data):**
    - `origin_image` (파일): 원본 JPEG/PNG 이미지
    - `sam_mask` (파일): 마스크 PNG 이미지
    - `trajectory_data` (파일): `[[x,y], [x,y], ...]` 형식의 JSON 파일
    - `frame_number` (폼 데이터, 선택): 생성할 프레임 수 (기본값: 14)

    **출력 (application/json):**
    - `job_id`와 `status_url`을 즉시 반환합니다.
    """
    
    if PIPELINE is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "서버가 준비되지 않았습니다. (모델 로딩 실패)"})

    # 1. 고유한 Job ID와 작업 폴더 생성
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 2. 업로드된 파일들을 고유한 작업 폴더에 저장 (API 명세서에 맞게)
    image_path = os.path.join(job_dir, "origin_image.jpg") # (확장자는 편의상 고정)
    mask_path = os.path.join(job_dir, "mask.png")
    traj_path = os.path.join(job_dir, "trajectory.json")

    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(origin_image.file, f)
        with open(mask_path, "wb") as f:
            shutil.copyfileobj(sam_mask.file, f)
        with open(traj_path, "wb") as f:
            shutil.copyfileobj(trajectory_data.file, f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"파일 저장 실패: {e}"})
    finally:
        origin_image.file.close()
        sam_mask.file.close()
        trajectory_data.file.close()

    # 3. DB에 '처리중' 상태 등록
    job_db[job_id] = {"status": "processing"}

    # 4. 백그라운드 작업 시작
    background_tasks.add_task(
        run_generation_task, 
        job_id, 
        job_dir, 
        image_path, 
        mask_path, 
        traj_path,
        frame_number # API로 받은 frame_number 전달
    )

    # 5. 사용자에게 Job ID와 상태 확인 URL 즉시 반환
    return {
        "status": "processing",
        "job_id": job_id,
        "status_url": f"/status/{job_id}"
    }


@app.get("/status/{job_id}", summary="작업 상태 확인")
async def get_job_status(job_id: str):
    """
    `POST /generate`에서 받은 `job_id`로 작업의 현재 상태를 확인합니다.

    **출력 (application/json):**
    - `status`: "processing", "complete", "error" 중 하나
    - (완료 시) `video_url`: 생성된 비디오의 다운로드 URL
    - (에러 시) `message`: 에러 메시지
    """
    status = job_db.get(job_id)
    if not status:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Job ID를 찾을 수 없습니다."})
    
    return status

@app.get("/", summary="API 상태 확인")
async def root():
    """
    API 서버가 살아있는지 확인합니다.
    """
    return {"message": "Motion Generation API가 실행 중입니다.", "model_loaded": PIPELINE is not None}


if __name__ == "__main__":
    # 포트 번호를 8001로 변경
    PORT_NUMBER = 8001 
    
    print(f"FastAPI 서버를 http://0.0.0.0:{PORT_NUMBER} 에서 시작합니다.")
    print(f"API 문서는 http://0.0.0.0:{PORT_NUMBER}/docs 에서 확인하세요.")
    uvicorn.run(app, host="0.0.0.0", port=PORT_NUMBER) # <--- 수정 완료
