# 이 파일은 demo.py의 모든 AI 로직과 헬퍼 함수를 포함
# (1/3) demo.py에서 가져온 모든 import
#
import os
import torch
import datetime
import numpy as np
from PIL import Image
# 'pipeline.pipeline_svd_DragAnything'이 실제 경로와 맞는지 확인 필요
from pipeline.pipeline_svd_DragAnything import StableVideoDiffusionPipeline
from models.DragAnything import DragAnythingSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as T
import torch.nn.functional as F
# 'utils.dift_util'이 실제 경로와 맞는지 확인 필요
from utils.dift_util import DIFT_Demo, SDFeaturizer
from torchvision.transforms import PILToTensor
import json

#
# (2/3) demo.py에서 가져온 모든 헬퍼 함수들 (수정 없이 그대로 복사)
#
def save_gifs_side_by_side(batch_output, validation_control_images,output_folder,name = 'none', target_size=(512 , 512),duration=200):
    # (demo.py의 원본 코드와 동일)
    flattened_batch_output = batch_output
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img,target_size=target_size) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    for idx, image_list in enumerate([validation_control_images, flattened_batch_output]):
        gif_path = os.path.join(output_folder, f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)
    def combine_gifs_side_by_side(gif_paths, output_path):
        print(gif_paths)
        gifs = [Image.open(gif) for gif in gif_paths]
        frames = []
        for frame_idx in range(gifs[0].n_frames):
            combined_frame = None
            for gif in gifs:
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)
        print(gifs[0].info['duration'])
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    combined_gif_path = os.path.join(output_folder, f"combined_frames_{name}_{timestamp}.gif")
    combine_gifs_side_by_side(gif_paths, combined_gif_path)
    for gif_path in gif_paths:
        os.remove(gif_path)
    return combined_gif_path

def validate_and_convert_image(image, target_size=(512 , 512)):
    # (demo.py의 원본 코드와 동일)
    if image is None:
        print("Encountered a None image")
        return None
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in [1, 3]: 
            if image.shape[0] == 1: 
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    return image

def create_image_grid(images, rows, cols, target_size=(512 , 512)):
    # (demo.py의 원본 코드와 동일)
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]
    if not valid_images:
        print("No valid images to create a grid")
        return None
    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))
    return grid

def tensor_to_pil(tensor):
    # (demo.py의 원본 코드와 동일)
    if len(tensor.shape) == 4: 
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else: 
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # (demo.py의 원본 코드와 동일)
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    validation_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images]
    validation_control_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_control_images]
    flattened_batch_output = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output]
    validation_images = [img for sublist in validation_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    validation_control_images = [img for sublist in validation_control_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_batch_output = [img for sublist in flattened_batch_output for img in (sublist if isinstance(sublist, list) else [sublist])]
    combined_frames = validation_images + validation_control_images + flattened_batch_output
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols
    grid = create_image_grid(combined_frames, rows, cols, target_size=(512, 512))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):
    # (demo.py의 원본 코드와 동일)
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"} 
    def frame_number(filename):
        matches = re.findall(r'\d+', filename) 
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2]) 
            return int(matches[-1]) 
        return float('inf') 
    sorted_files = sorted(os.listdir(folder), key=frame_number)
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)
    return images

def gen_gaussian_heatmap(imgSize=200):
    # (demo.py의 원본 코드와 동일)
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)
    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))
    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)
    return isotropicGrayscaleImage

def infer_model(model, image):
    # (demo.py의 원본 코드와 동일)
    transform = T.Compose([
        T.Resize((196, 196)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).cuda()
    cls_token = model(image, is_training=False)
    return cls_token

def find_largest_inner_rectangle_coordinates(mask_gray):
    # (demo.py의 원본 코드와 동일)
    refine_dist = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(refine_dist)
    radius = int(maxVal)
    return maxLoc, radius

def get_ID(images_list,masks_list,dinov2):
    # (demo.py의 원본 코드와 동일)
    ID_images = []
    image = images_list
    mask = masks_list
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    image = image * mask
    image = image[y:y+h,x:x+w]
    image = Image.fromarray(image).convert('RGB')
    img_embedding = infer_model(dinov2, image)
    return img_embedding

def get_dift_ID(feature_map,mask):
    # (demo.py의 원본 코드와 동일)
    new_feature = []
    non_zero_coordinates = np.column_stack(np.where(mask != 0))
    for coord in non_zero_coordinates:
        new_feature.append(feature_map[:, coord[0], coord[1]])
    stacked_tensor = torch.stack(new_feature, dim=0)
    average_pooled_tensor = torch.mean(stacked_tensor, dim=0)
    return average_pooled_tensor

def extract_dift_feature(image, dift_model):
    # (demo.py의 원본 코드와 동일)
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.open(image).convert('RGB')
    prompt = ''
    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    dift_feature = dift_model.forward(img_tensor, prompt=prompt, up_ft_index=3,ensemble_size=8)
    return dift_feature

def get_condition(target_size=(512 , 512), original_size=(512 , 512), args="", first_frame=None, is_mask = False, side=20,model_id=None):
    # (demo.py의 원본 코드와 동일)
    # (API 사용을 위해 약간 수정됨: args["validation_image"]는 이제 demo.json이 있는 '폴더' 경로)
    images = []
    vis_images = []
    heatmap = gen_gaussian_heatmap()

    original_size = (original_size[1],original_size[0])
    size = (target_size[1],target_size[0])
    latent_size = (int(target_size[1]/8), int(target_size[0]/8))

    dift_model = SDFeaturizer(sd_id=model_id)
    keyframe_dift = extract_dift_feature(first_frame, dift_model=dift_model)

    ID_images=[]
    ids_list={}

    # API용으로 수정: args["validation_image"]는 job_dir이 됨.
    json_path = os.path.join(args["validation_image"], "demo.json") 
    with open(json_path, 'r') as json_file:
        trajectory_json = json.load(json_file)

    mask_list = []
    trajectory_list = []
    radius_list = []

    for index in trajectory_json:
        ann = trajectory_json[index]
        mask_name = ann["mask_name"]
        trajectories = ann["trajectory"]
        trajectories = [[int(i[0]/original_size[0]*size[0]),int(i[1]/original_size[1]*size[1])] for i in trajectories]
        trajectory_list.append(trajectories)

        # API용으로 수정: 마스크 경로를 job_dir에서 읽음
        mask_path = os.path.join(args["validation_image"], mask_name)
        first_mask = (cv2.imread(mask_path)/255).astype(np.uint8)
        first_mask = cv2.cvtColor(first_mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mask_list.append(first_mask)

        mask_322 = cv2.resize(first_mask.astype(np.uint8),(int(target_size[1]), int(target_size[0])))
        _, radius = find_largest_inner_rectangle_coordinates(mask_322)
        radius_list.append(radius)
    
    # (get_condition의 나머지 로직은 demo.py와 동일)
    viss = 0
    if viss:
        mask_list_vis = [cv2.resize(i,(int(target_size[1]), int(target_size[0]))) for i in mask_list]
        vis_first_mask = show_mask(cv2.resize(np.array(first_frame).astype(np.uint8),(int(target_size[1]), int(target_size[0]))), mask_list_vis)
        vis_first_mask = cv2.cvtColor(vis_first_mask, cv2.COLOR_BGR2RGB)
        cv2.imwrite("test.jpg",vis_first_mask)
        assert False

    for idxx,point in enumerate(trajectory_list[0]):
        new_img = np.zeros(target_size, np.uint8)
        vis_img = new_img.copy()
        ids_embedding = torch.zeros((target_size[0], target_size[1], 320))

        if idxx>= args["frame_number"]:
            break

        for cc,(mask,trajectory,radius) in enumerate(zip(mask_list,trajectory_list,radius_list)):
            center_coordinate = trajectory[idxx]
            trajectory_ = trajectory[:idxx]
            side = min(radius,50)
            if idxx == 0:
                mask_32 = cv2.resize(mask.astype(np.uint8),latent_size)
                if len(np.column_stack(np.where(mask_32 != 0)))==0:
                    continue
                ids_list[cc] = get_dift_ID(keyframe_dift[0],mask_32)
                id_feature = ids_list[cc]
            else:
                if cc not in ids_list: 
                    continue
                id_feature = ids_list[cc]
            circle_img = np.zeros((target_size[0], target_size[1]), np.float32)
            circle_mask = cv2.circle(circle_img, (center_coordinate[0],center_coordinate[1]), side, 1, -1)
            y1 = max(center_coordinate[1]-side,0)
            y2 = min(center_coordinate[1]+side,target_size[0]-1)
            x1 = max(center_coordinate[0]-side,0)
            x2 = min(center_coordinate[0]+side,target_size[1]-1)
            if x2-x1>3 and y2-y1>3:
                need_map = cv2.resize(heatmap, (x2-x1, y2-y1))
                new_img[y1:y2,x1:x2] = need_map.copy()
                if cc>=0:
                    vis_img[y1:y2,x1:x2] = need_map.copy()
                    if len(trajectory_) == 1:
                        vis_img[trajectory_[0][1],trajectory_[0][0]] = 255
                    else:
                        for itt in range(len(trajectory_)-1):
                            cv2.line(vis_img,(trajectory_[itt][0],trajectory_[itt][1]),(trajectory_[itt+1][0],trajectory_[itt+1][1]),(255,255,255),3)
            non_zero_coordinates = np.column_stack(np.where(circle_mask != 0))
            for coord in non_zero_coordinates:
                ids_embedding[coord[0], coord[1]] = id_feature[0]

        ids_embedding = F.avg_pool1d(ids_embedding, kernel_size=2, stride=2)
        img = new_img
        if len(img.shape) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        images.append(pil_img)
        vis_images.append(Image.fromarray(vis_img))
        ID_images.append(ids_embedding)
    return images,ID_images,vis_images

def convert_list_bgra_to_rgba(image_list):
    # (demo.py의 원본 코드와 동일)
    rgba_images = []
    for image in image_list:
        if image.mode == 'RGBA' or image.mode == 'BGRA':
            b, g, r, a = image.split()
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))
        rgba_images.append(converted_image)
    return rgba_images

def show_mask(image, masks, random_color=False):
    # (demo.py의 원본 코드와 동일)
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
        h, w = mask.shape[:2]
        color_a = np.concatenate([np.random.random(3)*255], axis=0)
        mask_image = mask.reshape(h, w, 1) * color_a.reshape(1, 1, -1)
    else:
        h, w = masks[0].shape[:2]
        mask_image = 0
        for idx,mask in enumerate(masks):
            if idx!=1 and idx!=0:
                continue
            color = np.concatenate([np.random.random(3)*255], axis=0)
            mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1) + mask_image
    return (np.array(image).copy()*0.4+mask_image*0.6).astype(np.uint8)


#
# (3/3) API 서버를 위해 새로 추가된 함수들
#
def save_frames_as_video(frames, output_path, fps=10):
    """
    PIL Image 프레임 리스트를 MP4 비디오 파일로 저장합니다.
    (GIF 대신 MP4가 API 응답에 더 적합합니다.)
    """
    if not frames:
        print("저장할 프레임이 없습니다.")
        return

    # 첫 번째 프레임으로 비디오 라이터 설정
    try:
        first_frame = np.array(frames[0].convert('RGB'))
    except Exception as e:
        print(f"첫 번째 프레임 변환 실패: {e}")
        return
        
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 코덱
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # PIL Image (RGB) -> OpenCV (BGR)
        try:
            frame_bgr = cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        except Exception as e:
            print(f"프레임 변환 또는 쓰기 실패: {e}")
            continue # 한 프레임이 실패해도 계속 진행

    video.release()
    print(f"Video saved to {output_path}")


def load_pipeline(args):
    """
    모델과 파이프라인을 로드하여 반환합니다. 
    (API 서버 시작 시 1회만 호출됩니다.)
    
    (demo.py의 'if __name__ == "__main__":'에서 모델 로딩 부분만 가져옴)
    """
    print("Loading ControlNet (DragAnything)...")
    controlnet = DragAnythingSDVModel.from_pretrained(args["DragAnything"])
    
    print("Loading UNet...")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"], subfolder="unet")
    
    print("Loading SVD Pipeline...")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(args["pretrained_model_name_or_path"], controlnet=controlnet, unet=unet)
    
    pipeline.enable_model_cpu_offload() # 메모리 절약을 위해 GPU offload 활성화
    print("--- Pipeline loaded successfully ---")
    return pipeline


def run_inference_on_demand(
    pipeline,          # 로드된 파이프라인 객체
    pipeline_args,     # "model_DIFT", "frame_number" 등이 포함된 dict
    job_dir,           # 이 작업의 고유 폴더 (예: api_jobs/job-123)
    origin_image_path, # 원본 이미지 경로 (API 입력 1)
    mask_path,         # 마스크 이미지 경로 (API 입력 2)
    trajectory_path    # 궤적 JSON 파일 경로 (API 입력 3)
):
    """
    API 요청 1건을 처리하기 위해 demo.py의 핵심 로직(루프 1회)을 실행합니다.
    
    (demo.py의 'if __name__ == "__main__":'에서 루프 부분만 가져옴)
    """
    
    # 1. API 입력을 get_condition()이 읽을 수 있는 형식으로 변환합니다.
    #    (API 입력 3: trajectory_path -> demo.json)
    try:
        with open(trajectory_path, 'r') as f:
            trajectory_list = json.load(f)
            # API 명세서에 따라 trajectory_data가 [[x,y],...] 리스트여야 함
            if not isinstance(trajectory_list, list):
                 raise ValueError("궤적 데이터는 [x, y] 좌표의 리스트여야 합니다.")
    except Exception as e:
        raise ValueError(f"Trajectory 파일({trajectory_path}) 로드 실패: {e}")

    # (API 입력 2: mask_path -> demo.json)
    # get_condition이 job_dir에서 마스크 파일을 찾을 수 있도록 파일명 설정
    mask_name = os.path.basename(mask_path) 
    
    # get_condition이 읽을 임시 demo.json 생성
    temp_demo_json_data = {
        "0": { # 하나의 객체만 처리한다고 가정
            "mask_name": mask_name,
            "trajectory": trajectory_list
        }
    }
    
    temp_demo_json_path = os.path.join(job_dir, "demo.json")
    with open(temp_demo_json_path, 'w') as f:
        json.dump(temp_demo_json_data, f)

    # 2. demo.py와 동일하게 입력을 준비합니다.
    # (API 입력 1: origin_image_path)
    validation_image = Image.open(origin_image_path).convert('RGB')
    original_width, original_height = validation_image.size
    
    # API용 args를 설정합니다. (demo.py의 args를 대체)
    api_args = pipeline_args.copy()
    api_args["validation_image"] = job_dir #! get_condition이 job_dir 안의 demo.json과 마스크를 읽도록 설정
    
    target_height = api_args["height"]
    target_width = api_args["width"]
    num_frames = api_args["frame_number"] # 사용할 프레임 수

    validation_image_resized = validation_image.resize((target_width, target_height))

    # 3. get_condition()을 호출하여 컨트롤 맵 생성
    print(f"[Job {job_dir}] get_condition 호출 중...")
    validation_control_images, ids_embedding, vis_images = get_condition(
        target_size=(target_height, target_width),
        original_size=(original_height, original_width),
        args=api_args,
        first_frame=validation_image_resized,
        side=100,
        model_id=api_args["model_DIFT"]
    )
    
    ids_embedding = torch.stack(ids_embedding, dim=0).permute(0, 3, 1, 2)
    
    # 궤적 길이와 설정된 프레임 수 중 더 짧은 것을 사용
    num_frames_to_generate = min(len(validation_control_images), num_frames)
    if num_frames_to_generate == 0:
        raise ValueError("생성할 유효한 프레임(궤적)이 없습니다.")

    # 4. 파이프라인(추론) 실행
    print(f"[Job {job_dir}] 파이프라인 실행 중... (총 {num_frames_to_generate} 프레임)")
    video_frames_list = pipeline(
        validation_image_resized, 
        validation_control_images[:num_frames_to_generate], 
        decode_chunk_size=8,
        num_frames=num_frames_to_generate,
        motion_bucket_id=180,
        controlnet_cond_scale=1.0,
        height=target_height,
        width=target_width,
        ids_embedding=ids_embedding[:num_frames_to_generate]
    ).frames

    # video_frames_list는 [[img1, img2, ...]] 형태의 중첩 리스트임
    generated_frames = video_frames_list[0] 

    # 5. 결과를 MP4 비디오로 저장
    output_video_dir = os.path.join(job_dir, "output")
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, "generated_video.mp4")
    
    save_frames_as_video(generated_frames, output_video_path, fps=10) # 10 FPS로 저장 (조절 가능)

    print(f"[Job {job_dir}] 비디오 생성 완료: {output_video_path}")
    
    return output_video_path
