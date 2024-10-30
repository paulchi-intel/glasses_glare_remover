# 眼鏡反光消除系統

這是一個基於 Streamlit 開發的網頁應用程式，用於處理影片中的眼鏡反光問題。系統使用電腦視覺技術自動偵測並標記眼鏡區域。

## 功能特點

- 支援上傳 MP4、AVI、MOV 格式的影片檔案
- 自動偵測人臉和眼睛區域
- 即時顯示處理進度和效果預覽
- 可調整的處理參數
- 保留原始影片的音訊
- 支援處理後影片的下載

## 技術架構

### 核心技術
- **前端框架**: Streamlit
- **視覺處理**: OpenCV, dlib
- **影片處理**: MoviePy
- **人臉偵測**: dlib 人臉偵測器
- **特徵點偵測**: 68點人臉特徵偵測

### 主要元件

1. **人臉偵測模組**
   - 使用 dlib 的前置人臉偵測器
   - 使用 68 點特徵偵測器定位眼睛位置

2. **影片處理模組**
   - 支援逐幀處理
   - 保留原始音訊
   - 處理進度即時顯示

3. **使用者介面**
   - 參數調整側邊欄
   - 即時預覽視窗
   - 進度指示器

## 系統需求

- Python 3.9
- OpenCV
- dlib
- Streamlit
- MoviePy

## 安裝說明

1. 複製專案：
   ```bash
   git clone https://github.com/yourusername/glasses-glare-remover.git
   cd glasses-glare-remover
   ```

2. 建立並啟動虛擬環境：
   ```bash
   python -m venv .venv
   # Windows：
   .venv\Scripts\activate
   # macOS/Linux：
   source .venv/bin/activate
   ```

3. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```

4. 下載必要的模型檔案：
   - 從 [dlib 模型下載頁面](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) 下載特徵點偵測器
   - 解壓縮並將 `shape_predictor_68_face_landmarks.dat` 放置在 `bin` 目錄下

## 使用說明

1. 啟動應用程式：
   ```bash
   streamlit run app.py
   ```

2. 透過網頁介面：
   - 上傳影片檔案
   - 調整處理參數
   - 點擊「處理影片」按鈕
   - 下載處理後的影片

### 可調整參數

- **高斯模糊核心大小**: 控制模糊效果的強度（範圍：3-21，建議值：9）
- **形態學運算核心大小**: 控制區域擴張/收縮的程度（範圍：3-11，建議值：5）
- **反光區域擴張次數**: 控制處理區域的大小（範圍：0-5，建議值：2）
- **修復半徑**: 控制修復演算法的作用範圍（範圍：1-20，建議值：10）
- **修復方法**: 選擇不同的修復演算法
  - Telea: 適合處理小區域
  - Navier-Stokes: 適合處理較大區域

## 專案結構
```
.
├── app.py
├── bin/
│ └── shape_predictor_68_face_landmarks.dat
├── input/
│ └── Glasses Glare Remover.mp4
├── requirements.txt
└── README.md