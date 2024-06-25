import os
import numpy as np
from ultralytics import YOLO
import torch
import cv2
import tensorflow as tf



# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for g in gpus:
#     if g:
#         try:
#             tf.config.experimental.set_virtual_device_configuration(
#                 g,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#         except RuntimeError as e:
#             print(e)
# Check video length
# class_name = 'Normal_Wess'


# # Check video length
# print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (2, 3 두개 사용하므로)
# print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)

def is_video_lower_one_hour(video_path, check_hour = 1):
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)

    # 총 프레임 수
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 프레임 레이트
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # 비디오의 총 길이 계산 (초 단위)
    video_length_seconds = total_frames / frame_rate

    # 비디오의 총 길이를 시간으로 변환 (시, 분, 초)
    video_length_hours = video_length_seconds / 3600
    # 비디오 닫기
    cap.release()
    # 1시간을 넘는지 확인하는 if 문
    if video_length_hours <= check_hour and frame_rate == 30:
        return True
    return False


# Load a model




model = YOLO('/home/sm32289/Cat_Pose/KeypointDetection_Yolo/Model/train6/weights/best.pt')  # load a pretrained model (recommended for training)\
fold = ['cat eating', 'cat sitting', 'cat sleeping', 'cat_running', 'cat_walking']



num_features = 30

for class_name in fold :
    # 폴더 내의 모든 동영상 파일 목록 가져오기
    video_folder = os.path.join('/home/sm32289/Cat_Pose/KeypointDetection_Yolo/Test_video_30/Normal_wess',class_name)
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    remove_videos = []
    change_videos = []
    predict_dir = '/home/sm32289/Cat_Pose/KeypointDetection_Yolo/predict2/Normal_wess' 
    if os.path.exists(os.path.join(predict_dir, class_name,'video')) == False:
        os.makedirs(os.path.join(predict_dir, class_name,'video'))
    predict_video_path = os.path.join(predict_dir, class_name,'video') 
    if os.path.exists(os.path.join(predict_dir, class_name,'keypoints')) == False:
        os.makedirs(os.path.join(predict_dir, class_name,'keypoints'))
    predict_keypoints_path =os.path.join(predict_dir, class_name,'keypoints')

    # 모델 테스트 및 결과 저장
    for video_file in video_files:
        # 모델을 사용하여 동영상 처리 및 결과 얻기
        video_path = os.path.join(video_folder, video_file)
        if is_video_lower_one_hour(video_path):
            # 결과 동영상 파일 이름 생성
            output_video_file = os.path.splitext(video_file)[0]
            key_save_dir = os.path.join(predict_keypoints_path,output_video_file)

            results = model(source=video_path, conf=0.8, save=True, project=predict_video_path, name = output_video_file, device = [0,1,2,3])
            print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (2, 3 두개 사용하므로)
            print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)         
            save_dir = os.path.join(predict_video_path , output_video_file)
            if not os.path.exists(save_dir):
                # 결과 동영상을 .avi에서 .mp4로 변환
                output_video_path = os.path.join(save_dir, output_video_file +'.avi')
                output_mp4_path = os.path.join(save_dir,output_video_file + '.mp4')
                remove_videos.append(output_video_path)
                change_videos.append(output_mp4_path)
                if not os.path.exists(output_mp4_path):
                    os.system(f'ffmpeg -i "{output_video_path}" -c:v libx264 -crf 23 -c:a aac -strict -2 "{output_mp4_path}"')
                    cnt = 0
                    # 결과 키포인트 데이터 저장( == 한마리의 고양이만 있을 경우만 데이터로 저장, 30FPS로 이루어져 있음)
                    for i in range(len(results)):
                        bbox_xyxy = results[i].boxes.xyxy.flatten().cpu().numpy()  # x1, y1, x2, y2
                        keypoints_xy = results[i].keypoints.xy.flatten().cpu().numpy()  
                        if bbox_xyxy.size != 0 and len(keypoints_xy) == num_features:           
                        # bbox 크기 가져오기
                            bbox_width = bbox_xyxy[2] - bbox_xyxy[0]
                            bbox_height = bbox_xyxy[3] - bbox_xyxy[1]
                            for i in range(0,30,2):
                                keypoints_xy[i] = (keypoints_xy[i] - bbox_xyxy[0])/bbox_width
                                keypoints_xy[i+1] =  (keypoints_xy[i+1] - bbox_xyxy[1])/bbox_height
                            keypoints_xy[keypoints_xy < 0] = 0
                            keypoints_xy[keypoints_xy > 1] = 1
                                
                            f = bbox_xyxy.tolist() + keypoints_xy.tolist()
                            keypoints_file_path = os.path.join(key_save_dir, f'{cnt}.npy')
                            cnt+=1
                            os.makedirs(key_save_dir,exist_ok=True)
                            np.save(keypoints_file_path, f) 
                        
#.avi영상 지우기
for re in remove_videos:
    os.remove(re)