import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
import joblib 
import cv2 # Thêm OpenCV để vẽ Grad-CAM

# Import class ConvNet từ file của Thái
from modules.transfer_learning import ConvNet

st.set_page_config(page_title="Chẩn đoán ảnh y tế - Nhóm 9", layout="wide")

st.title("Phân loại ảnh y tế với ResNet50 + AAPSO + k-NN")
st.markdown("📍 **Đồ án Khai thác dữ liệu - Nhóm 9**")

DATASET_CONFIG = {
    "Chest (X-quang phổi)": {
        "weights": "weights/best_chest.pth", 
        "mask": "weights/mask_best_chest.npy",
        "knn": "weights/knn_best_chest.pkl",
        "num_classes": 2, 
        "labels":["Bình thường (Normal)", "Viêm phổi (Pneumonia)"]
    },
    "Breast (Siêu âm tuyến vú)": {
        "weights": "weights/best_breast.pth", 
        "mask": "weights/mask_best_breast.npy",
        "knn": "weights/knn_best_breast.pkl",
        "num_classes": 2, 
        "labels": ["Lành tính/Bình thường (Benign)", "Ác tính (Malignant)"]
    },
    "Derma (Bệnh ngoài da)": {
        "weights": "weights/best_derma.pth", 
        "mask": "weights/mask_best_derma.npy",
        "knn": "weights/knn_best_derma.pkl",
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

st.sidebar.title("⚙️ Cài đặt cấu hình")
dataset_choice = st.sidebar.selectbox("Chọn loại bộ dữ liệu:", list(DATASET_CONFIG.keys()))
config = DATASET_CONFIG[dataset_choice]

@st.cache_resource
def load_model(weights_path, num_classes):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                base_model = models.resnet50(weights=None)
            except:
                base_model = models.resnet50(pretrained=False)
                
        model = ConvNet(base_model, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        return None

model = load_model(config["weights"], config["num_classes"])

# --- HÀM TẠO GRAD-CAM ---
def generate_gradcam(model, input_tensor, original_image):
    model.eval()
    
    # Lấy layer4 của ResNet50 (Lớp tích chập cuối cùng để trích xuất đặc trưng không gian)
    target_layer = model.base_model[-2] 

    feature_maps = []
    gradients = []

    # Đăng ký hook để "bắt" dữ liệu lúc model phân tích
    def forward_hook(module, input, output):
        feature_maps.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    # Chạy model
    model.zero_grad()
    _, outputs = model(input_tensor)
    
    # Lấy class có điểm số cao nhất để tính đạo hàm ngược
    target_class = outputs.argmax(dim=1).item()
    score = outputs[0, target_class]
    score.backward()

    # Tính toán Heatmap
    grad = gradients[0].cpu().data.numpy()[0]
    fmap = feature_maps[0].cpu().data.numpy()[0]
    weights_cam = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights_cam):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0) # Tương đương ReLU
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Gỡ hook
    handle_fw.remove()
    handle_bw.remove()

    # Phủ màu tản nhiệt lên ảnh gốc
    img_np = np.array(original_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Trộn ảnh (0.4 heatmap + 0.6 ảnh gốc)
    superimposed_img = np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6
    superimposed_img = np.uint8(superimposed_img / np.max(superimposed_img) * 255)

    return Image.fromarray(superimposed_img)


# --- GIAO DIỆN CHÍNH ---
st.write(f"### Đang chạy Pipeline: `{dataset_choice}`")
uploaded_file = st.file_uploader("Tải lên ảnh y tế (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2, col3 = st.columns([1, 1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption='Ảnh Gốc', use_container_width=True)
    
    if st.button("🚀 Tiến hành phân loại", use_container_width=True):
        if model is None:
            st.error("❌ Không tìm thấy file trọng số (.pth). Vui lòng kiểm tra lại!")
        else:
            with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
                try:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = transform(image).unsqueeze(0) 
                    
                    # 1. Vẽ Grad-CAM
                    gradcam_img = generate_gradcam(model, input_tensor, image)
                    with col2:
                        st.image(gradcam_img, caption='Vùng tập trung đặc trưng (Grad-CAM)', use_container_width=True)
                    
                    # 2. Feedforward lấy Vector 512
                    with torch.no_grad():
                        features_512, _ = model(input_tensor)
                    features_np = features_512.cpu().numpy()

                    # 3. Load Mask & k-NN
                    mask = np.load(config["mask"])
                    knn = joblib.load(config["knn"])

                    # 4. Lọc đặc trưng & Phân loại
                    selected_features = features_np[:, mask == 1]
                    num_selected = int(np.sum(mask))

                    predicted_class_idx = knn.predict(selected_features)[0]
                    probabilities = knn.predict_proba(selected_features)[0]
                    confidence = probabilities[predicted_class_idx]
                    predicted_label = config["labels"][predicted_class_idx]
                    
                    # 5. In kết quả
                    with col3:
                        st.success(f"**Kết quả chẩn đoán:**\n\n{predicted_label}")
                        st.info(f"**Độ tin cậy (k-NN):** {confidence*100:.2f}%")
                        
                        with st.expander("📊 Xem chi tiết Pipeline"):
                            st.write(f"**1. ResNet50:** Trích xuất `512` đặc trưng không gian.")
                            st.write(f"**2. AAPSO:** Loại bỏ nhiễu, giữ lại `{num_selected}` đặc trưng tinh túy (Giảm {(1 - num_selected/512)*100:.2f}%).")
                            st.write(f"**3. k-NN:** Tính toán khoảng cách trên không gian {num_selected} chiều.")
                        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: Cần có đủ 3 file .pth, .npy, .pkl trong thư mục weights. Chi tiết: {e}")
