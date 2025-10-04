# Image_AI_API
## 환경 구축  
### (1) 환경 생성  
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


