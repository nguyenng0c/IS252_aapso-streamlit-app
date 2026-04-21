import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import warnings

# Import class ConvNet từ file của Thái
from modules.transfer_learning import ConvNet

st.set_page_config(page_title="MedMNIST v2 Classification", layout="wide")

st.title("Phân loại ảnh y tế với ResNet50 + AAPSO + k-NN")
st.markdown("📍 **Đồ án Khai thác dữ liệu - Nhóm 9**")

# --- 1. CẤU HÌNH DATASET ---
# Từ điển lưu số lớp và nhãn (Labels) dựa trên MedMNIST
DATASET_CONFIG = {
    "Chest (X-quang phổi)": {
        "weights": "weights/best_chest.pth", 
        "num_classes": 2, 
        "labels":["Bình thường (Normal)", "Viêm phổi (Pneumonia)"]
    },
    "Breast (Siêu âm tuyến vú)": {
        "weights": "weights/best_breast.pth", 
        "num_classes": 2, 
        "labels": ["Lành tính/Bình thường (Benign)", "Ác tính (Malignant)"]
    },
    "Derma (Bệnh ngoài da)": {
        "weights": "weights/best_derma.pth", 
        "num_classes": 7, 
        "labels":["Actinic keratoses (Dày sừng quang hóa)", 
                   "Basal cell carcinoma (Ung thư biểu mô tế bào đáy)", 
                   "Benign keratosis-like (Tổn thương sừng lành tính)", 
                   "Dermatofibroma (U xơ da)", 
                   "Melanoma (Khối u ác tính)", 
                   "Melanocytic nevi (Nốt ruồi hắc tố)", 
                   "Vascular lesions (Tổn thương mạch máu)"]
    }
}

# --- 2. SIDEBAR CHỌN MODEL ---
st.sidebar.title("⚙️ Cài đặt cấu hình")
dataset_choice = st.sidebar.selectbox("Chọn loại bộ dữ liệu:", list(DATASET_CONFIG.keys()))
config = DATASET_CONFIG[dataset_choice]

# --- 3. HÀM LOAD MODEL ---
@st.cache_resource
def load_model(weights_path, num_classes):
    try:
        # Xử lý tương thích phiên bản torchvision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                base_model = models.resnet50(weights=None)
            except:
                base_model = models.resnet50(pretrained=False)
                
        # Khởi tạo mô hình ConvNet của nhóm
        model = ConvNet(base_model, num_classes)
        # Load file weights (.pth)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        return None

# Gọi hàm load model
model = load_model(config["weights"], config["num_classes"])

# --- 4. XỬ LÝ UPLOAD & DỰ ĐOÁN ---
st.write(f"### Đang chạy mô hình: `{dataset_choice}`")
uploaded_file = st.file_uploader("Tải lên ảnh y tế (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    if st.button("🚀 Tiến hành phân loại", use_container_width=True):
        if model is None:
            st.error("❌ Không tìm thấy file trọng số (weights). Vui lòng kiểm tra lại thư mục `weights/`!")
        else:
            with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
                try:
                    # Tiền xử lý ảnh (Resize 224x224 và Normalize theo chuẩn ImageNet)
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = transform(image).unsqueeze(0) # Thêm batch size = 1
                    
                    # Feedforward qua model
                    with torch.no_grad():
                        features_512, outputs = model(input_tensor)
                        
                        # Softmax để tính xác suất phần trăm
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        predicted_class_idx = torch.argmax(probabilities).item()
                        confidence = probabilities[predicted_class_idx].item()
                    
                    # Lấy nhãn kết quả
                    predicted_label = config["labels"][predicted_class_idx]
                    
                    # In kết quả ra màn hình bên phải
                    with col2:
                        st.success(f"**Kết quả chẩn đoán:** {predicted_label}")
                        st.info(f"**Độ tin cậy:** {confidence*100:.2f}%")
                        
                        with st.expander("Xem vector đặc trưng (ResNet50)"):
                            st.write(f"Kích thước vector: `{features_512.shape}`")
                            st.write(features_512.numpy())
                            
                        # Dành sẵn UI báo cho GV biết sẽ ráp AAPSO + k-NN vào sau
                        st.warning("🔄 **Giai đoạn tiếp theo:** Đưa vector 512 này qua AAPSO để chọn lọc đặc trưng, sau đó phân lớp bằng k-NN.")
                        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")