import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import logging
import moviepy.editor as mp
import dlib
import tempfile

logging.basicConfig(level=logging.INFO)

# 加載人臉檢測器和眼睛檢測器
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

print(f"Face cascade path: {face_cascade_path}")
print(f"Eye cascade path: {eye_cascade_path}")

if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path):
    raise FileNotFoundError("Cascade files not found. Please check the paths.")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    raise ValueError("Failed to load face cascade classifier")
if eye_cascade.empty():
    raise ValueError("Failed to load eye cascade classifier")

# 加載 Dlib 的人臉檢測器和特徵點檢測器
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join("bin", "shape_predictor_68_face_landmarks.dat")

# 添加檢查代碼
if not os.path.exists(predictor_path):
    st.error(f"找不到模型文件：{predictor_path}")
    st.markdown(f"""
    請確認模型文件是否存在於正確位置：
    - 當前尋找路徑：{os.path.abspath(predictor_path)}
    
    如果文件不存在，請：
    1. 從[這裡下載](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    2. 解壓文件
    3. 將 shape_predictor_68_face_landmarks.dat 放在 bin 目錄下
    """)
    st.stop()

predictor = dlib.shape_predictor(predictor_path)

def detect_and_correct_glare(image, blur_size, threshold, morph_size, dilate_iterations, inpaint_radius, inpaint_method):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)  # 使用 Dlib 檢測人臉
        
        if len(faces) == 0:
            logging.info("No faces detected in this frame")
            return image
        
        for face in faces:
            landmarks = predictor(gray, face)  # 獲取特徵點
            
            # 獲取眼睛的邊界
            left_eye = (landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(39).y)  # 左眼
            right_eye = (landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y)  # 右眼
            
            # 計算眼鏡區域，考慮修復半徑和擴張次數
            glasses_x = min(left_eye[0], right_eye[0]) - inpaint_radius - 10  # 向左擴展
            glasses_y = min(left_eye[1], right_eye[1]) - inpaint_radius - 10  # 向上擴展
            glasses_w = max(left_eye[2], right_eye[2]) - glasses_x + (inpaint_radius * 2) + 20  # 向右擴展
            glasses_h = max(left_eye[3], right_eye[3]) - glasses_y + (inpaint_radius * 2) + 20  # 向下擴展
            
            # 確保座標在有效範圍內
            glasses_x = max(0, glasses_x)
            glasses_y = max(0, glasses_y)
            glasses_w = min(glasses_w, image.shape[1] - glasses_x)
            glasses_h = min(glasses_h, image.shape[0] - glasses_y)
            
            # 標記眼鏡區域
            cv2.rectangle(image, 
                          (glasses_x, glasses_y), 
                          (glasses_x + glasses_w, glasses_y + glasses_h), 
                          (0, 255, 0), 2)  # 綠色標註眼鏡區域
            
            # 標記眼睛區域
            cv2.rectangle(image, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (255, 0, 0), 1)  # 藍色標註左眼
            cv2.rectangle(image, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (255, 0, 0), 1)  # 藍色標註右眼
        
        return image
    except Exception as e:
        logging.error(f"Error in detect_and_correct_glare: {str(e)}")
        return image

def process_frame(frame, blur_size, morph_size, dilate_iterations, inpaint_radius, inpaint_method):
    return detect_and_correct_glare(frame, blur_size, 0, morph_size, dilate_iterations, inpaint_radius, inpaint_method)

def process_video(input_bytes, blur_size, morph_size, dilate_iterations, inpaint_radius, inpaint_method, show_comparison=False):
    try:
        # 將輸入視頻數據轉換為臨時文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            temp_input.write(input_bytes)
            temp_input_path = temp_input.name

        cap = cv2.VideoCapture(temp_input_path)
        
        if not cap.isOpened():
            raise ValueError("無法打開視頻文件")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 創建臨時輸出文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            temp_output_path = temp_output.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                original_frame = frame.copy()
                processed_frame = process_frame(frame, blur_size, morph_size, dilate_iterations, inpaint_radius, inpaint_method)
                out.write(processed_frame)
                
                frame_count += 1
                progress = int(frame_count / total_frames * 100)
                progress_bar.progress(progress)
                
                if frame_count % 30 == 0:
                    if show_comparison:
                        comparison = np.hstack((original_frame, processed_frame))
                        frame_placeholder.image(comparison, channels="BGR", caption=f"處理進度: {progress}% (左: 原始, 右: 處理後)")
                    else:
                        frame_placeholder.image(processed_frame, channels="BGR", caption=f"處理進度: {progress}%")
            
            cap.release()
            out.release()

        # 使用 moviepy 處理音頻
        video = mp.VideoFileClip(temp_output_path)
        original_audio = mp.VideoFileClip(temp_input_path).audio
        
        # 創建最終輸出的臨時文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as final_temp:
            final_output_path = final_temp.name
            final_video = video.set_audio(original_audio)
            final_video.write_videofile(final_output_path, codec="libx264")
        
        # 讀取處理後的視頻到內存
        with open(final_output_path, 'rb') as f:
            processed_video_data = f.read()
        
        # 清理所有臨時文件
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        os.unlink(final_output_path)
        
        progress_bar.empty()
        frame_placeholder.empty()
        
        return processed_video_data
            
    except Exception as e:
        st.error(f"處理視頻時發生錯誤: {str(e)}")
        logging.error(f"Error in process_video: {str(e)}")
        return None
    
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

st.title('眼鏡反光消除')

# 添加範例視頻的下載鏈接，不顯示標題
example_video_path = "input/Glasses Glare Remover.mp4"

if os.path.exists(example_video_path):
    with open(example_video_path, "rb") as example_file:
        st.download_button(
            label="下載範例視頻",
            data=example_file,
            file_name="Glasses_Glare_Remover.mp4",
            mime="video/mp4"
        )
else:
    st.error("範例視頻文件未找到")

uploaded_file = st.file_uploader("上傳視頻文件", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.sidebar.header("參數設置")
    blur_size = st.sidebar.slider("高斯模糊核大小", 3, 21, 9, step=2)
    morph_size = st.sidebar.slider("形態學操作核大小", 3, 11, 5, step=2)
    dilate_iterations = st.sidebar.slider("反光區域擴張次數", 0, 5, 2)
    inpaint_radius = st.sidebar.slider("修復半徑", 1, 20, 10)
    inpaint_method = st.sidebar.selectbox("修復方法", ["INPAINT_TELEA", "INPAINT_NS"], format_func=lambda x: "Telea" if x == "INPAINT_TELEA" else "Navier-Stokes")
    show_comparison = st.sidebar.checkbox("顯示原始和處理後的對比", value=True)

    if st.button("處理視頻"):
        try:
            # 直接讀取上傳文件的內容
            video_bytes = uploaded_file.read()
            
            processed_video_data = process_video(
                video_bytes,
                blur_size, 
                morph_size,
                dilate_iterations, 
                inpaint_radius,
                getattr(cv2, inpaint_method),
                show_comparison
            )
            
            if processed_video_data:
                # 顯示處理後的視頻
                st.video(processed_video_data)
                
                # 提供下載按鈕
                st.download_button(
                    label="下載處理後的視頻",
                    data=processed_video_data,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("視頻處理失敗")
            
        except Exception as e:
            st.error(f"處理過程中發生錯誤: {str(e)}")
            logging.error(f"Error during processing: {str(e)}")


