# Image_AI_API
## 주의 사항
inpaint_app.py 내 lora 파일 경로가 서버의 캐시 경로로 지정되어 있으므로, 
필요시 꼭 수정 요망.

## 환경 구축  
### (1) 이미지 생성/수정 환경 생성  
```bash
mkdir flux-storyboard-api
cd flux-storyboard-api

git clone https://github.com/2025-2-ITEC0402-001/Image_AI_API.git

conda env create -f environment.yml
```

```bash
conda activate flux_api_env

pip install git+https://github.com/huggingface/accelerate.git
pip install git+https://github.com/huggingface/diffusers.git
pip install "fastapi[all]"
```

### 영상 생성 API 관련 환경 생성  
```bash
cd /abr/coss33/proj2/DragAnything
conda activate motioni2v
pip install fastapi uvicorn python-multipart
```
  
## 실행  
### (1) API 실행  

- 이미지 생성 API 실행:
  ```bash
  uvicorn generate_app:app --host 0.0.0.0 --port 5000
  ```
    
- 이미지 수정 API 실행:
  ```bash
  uvicorn inpaint_app:app --host 0.0.0.0 --port 5001
  ```

- 영상 생성 API 실행:
  ```bash
  python fastapi_server.py
  ```
  서버가 `http://0.0.0.0:8001`에서 실행됩니다.
  API 문서는 `http://localhost:8001/docs`에서 실시간으로 확인할 수 있습니다.
     


## 테스트  
### (1) 이미지 생성:
```bash
curl -X POST http://localhost:5000/generate-storyboard -H "Content-Type: application/json" -d '{"prompt": "[MOVIE-SHOTS] a cute cat programmer writing code on a laptop, anime style"}' --output generated_image.png
```
<img width="1116" height="140" alt="image" src="https://github.com/user-attachments/assets/f8965a6a-ab5d-435e-b4d9-cbaf4ada2267" />   
<img width="256" height="384" alt="generated_image" src="https://github.com/user-attachments/assets/67d43851-28d2-4242-87cf-e2140831b6fc" />


   
### (2) 이미지 수정:  
```bash
curl -X POST http://localhost:5001/revise-storyboard -F "image=@generated_image.png" -F "revised_prompt=[MOVIE-SHOTS] a tired dog programmer falling asleep on a laptop, anime style" --output revised_image.png
```
<img width="1118" height="138" alt="image" src="https://github.com/user-attachments/assets/08c411bc-c2e7-43b1-b38a-d8ecbdf53223" />  
<img width="256" height="384" alt="reviesed_image" src="https://github.com/user-attachments/assets/4516bf51-47af-4a5d-9a6c-f59bba93d348" />  


### (3) 영상 생성: 

(1) 비디오 생성 요청 (포트 8001): POST /generate
- Input: multipart/form-data (원본 이미지, 마스크, 궤적 JSON)  
- Output: application/json (job_id 즉시 반환)  
- cURL 테스트: (테스트용 test_coords.json 파일이 DragAnything 루트에 있다고 가정)  
```bash
curl -X POST "http://localhost:8001/generate" \
 -F "origin_image=@data/Processed_VIPSeg_Demo/1002_3nW_Y_u1S08/demo.jpg" \
 -F "sam_mask=@data/Processed_VIPSeg_Demo/1002_3nW_Y_u1S08/mask_car_0.png" \
 -F "trajectory_data=@test_coords.json"
```  
- 성공 응답 (예시):
  ```JSON
  {
  "status": "processing",
  "job_id": "74c6b783-7944-4c45-aef5-7792549208a3",
  "status_url": "/status/74c6b783-7944-4c45-aef5-7792549208a3"
  }
  ```

(2) 작업 상태 확인 (포트 8001): `GET /status/{job_id}`  
- Input: job_id (Path Parameter)  
- Output: application/json (작업 상태)    
- cURL 테스트: (1단계에서 받은 job_id 사용)
```bash
# (완료될 때까지 클라이언트에서 10초 같은 주기 마다 반복 요청하면 됨)
curl http://localhost:8001/status/74c6b783-7944-4c45-aef5-7792549208a3
```
- 완료 시 응답 (예시):
  ```JSON
  {
  "status": "complete",
  "video_url": "/results/74c6b783-7944-4c45-aef5-7792549208a3/output/generated_video.mp4"
  }
  ```

(3) 비디오 다운로드:  
- 2단계에서 받은 video_url을 통해 파일을 다운로드합니다.  
- cURL 테스트:
```bash
curl -o my_generated_video.mp4 "http://localhost:8001/results/74c6b783-7944-4c45-aef5-7792549208a3/output/generated_video.mp4"
```

