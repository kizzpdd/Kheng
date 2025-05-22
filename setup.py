import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# --- 1. Hàm tiền xử lý Z-score ---
def z_score_normalize(spectrum):
    """Chuẩn hóa dữ liệu phổ bằng Z-score."""
    mean_val = np.mean(spectrum)
    std_dev = np.std(spectrum)
    if std_dev == 0:
        # Trả về mảng 0 nếu độ lệch chuẩn là 0 để tránh lỗi chia cho 0
        return np.zeros_like(spectrum)
    return (spectrum - mean_val) / std_dev

# --- 2. Hàm tăng cường dữ liệu (Data Augmentation) ---
def augment_spectrum(spectrum, scale_factor_range=(0.6, 1.4)):
    """
    Tăng cường phổ bằng cách nhân với một hệ số tỷ lệ ngẫu nhiên.
    """
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    return spectrum * scale_factor

# --- 3. Hàm đọc, tiền xử lý và tăng cường dữ liệu (từ code của bạn) ---
def load_and_preprocess_data_from_folders(base_folder_path, target_wavelength=640):
    """
    Đọc dữ liệu phổ từ các file .dpt trong các thư mục con,
    cắt phổ, chuẩn hóa, tăng cường và trả về dữ liệu đã xử lý cùng nhãn.
    """
    data = []
    raw_labels = [] # Sẽ chứa tên folder làm nhãn
    max_length = 0

    print(f"Đang quét thư mục: {base_folder_path}")
    if not os.path.isdir(base_folder_path):
        print(f"Lỗi: Thư mục gốc '{base_folder_path}' không tồn tại.")
        return None, None, None

    for folder_name in os.listdir(base_folder_path):
        sub_folder_path = os.path.join(base_folder_path, folder_name)
        if os.path.isdir(sub_folder_path):
            print(f"  Đang xử lý thư mục con: {folder_name}")
            for file_name in os.listdir(sub_folder_path):
                if file_name.endswith('.dpt'):
                    file_path = os.path.join(sub_folder_path, file_name)
                    try:
                        spectrum_raw = np.loadtxt(file_path, delimiter=',')
                        intensity = spectrum_raw[:, 1] # Cột 1 là cường độ
                        wave = spectrum_raw[:, 0]    # Cột 0 là bước sóng

                        # Tìm index cuối cùng trước 640nm bằng np.searchsorted
                        # np.searchsorted trả về index nơi 640.0 có thể được chèn vào
                        # để duy trì thứ tự tăng dần.
                        # Do đó, tất cả các giá trị trước index này sẽ nhỏ hơn 640.0
                        length_after_cut = np.searchsorted(wave, target_wavelength)
                        intensity_short = intensity[:length_after_cut]

                        # Chuẩn hóa dữ liệu bằng Z-score
                        normalized_values = z_score_normalize(intensity_short)

                        # Thêm phổ gốc đã chuẩn hóa
                        data.append(normalized_values)
                        raw_labels.append(folder_name) # Tên folder là nhãn
                        max_length = max(max_length, len(normalized_values))

                        # Data Augmentation - Tăng cường dữ liệu
                        # Tạo 2 phiên bản tăng cường từ phổ gốc
                        augmented_data_1 = augment_spectrum(normalized_values, scale_factor_range=(0.85, 0.95))
                        data.append(augmented_data_1)
                        raw_labels.append(folder_name)

                        augmented_data_2 = augment_spectrum(normalized_values, scale_factor_range=(1.05, 1.15))
                        data.append(augmented_data_2)
                        raw_labels.append(folder_name)

                    except Exception as e:
                        print(f"Lỗi khi đọc hoặc xử lý file {file_path}: {e}")
    
    if not data:
        print("Không tìm thấy dữ liệu phổ nào để xử lý.")
        return None, None, None

    # Padding sequences to the maximum length (zero-padding ở cuối)
    padded_data = []
    for spectrum in data:
        current_length = len(spectrum)
        if current_length < max_length:
            padding_length = max_length - current_length
            # Pad ở cuối với giá trị 0
            padded_spectrum = np.pad(spectrum, (0, padding_length), 'constant', constant_values=0)
        else:
            # Nếu phổ đã có độ dài bằng max_length (hoặc lớn hơn, nhưng không nên xảy ra)
            padded_spectrum = spectrum[:max_length] # Đảm bảo không vượt quá max_length
        padded_data.append(padded_spectrum)
    
    return np.array(padded_data), np.array(raw_labels), max_length

# --- Cấu hình đường dẫn thư mục gốc chứa các folder 'Pho_A', 'Pho_B', v.v. ---
# Ví dụ: nếu cấu trúc thư mục của bạn là:
# D:\Data\Fourier\Data_250502\
# ├── Pho_A/
# │   ├── spectrum1.dpt
# │   └── spectrum2.dpt
# └── Pho_B/
#     ├── spectrum3.dpt
#     └── spectrum4.dpt
# Thì folder_path sẽ là 'D:\Data\Fourier\Data_250502'
folder_path = r'D:\Data\Fourier\Data_250502' # Đặt đường dẫn của bạn ở đây

# --- 4. Tải và tiền xử lý toàn bộ dữ liệu ---
print("Đang tải, tiền xử lý và tăng cường dữ liệu...")
X_processed, y_raw_labels, max_spectrum_length = load_and_preprocess_data_from_folders(folder_path, target_wavelength=640)

if X_processed is None:
    print("Không thể tải dữ liệu để huấn luyện. Vui lòng kiểm tra đường dẫn và cấu trúc file.")
    exit()

print(f"Tổng số phổ đã tải và tăng cường: {len(X_processed)}")
print(f"Độ dài phổ tối đa sau cắt và pad: {max_spectrum_length}")

# Chuyển đổi nhãn string thành nhãn số (0, 1, ...)
le = LabelEncoder()
y = le.fit_transform(y_raw_labels)
class_names = le.classes_ # Lấy lại tên các lớp (ví dụ: 'Pho_A', 'Pho_B')
print(f"Các lớp được tìm thấy và mã hóa: {class_names}")

# --- 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra ---
# Sử dụng stratify=y để đảm bảo tỷ lệ các lớp trong tập train/test là đồng đều
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print(f"Số lượng mẫu huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu kiểm tra: {len(X_test)}")

# --- 6. Huấn luyện mô hình Support Vector Machine (SVM) ---
print("Đang huấn luyện mô hình SVM...")
# kernel='linear' thường là lựa chọn tốt cho dữ liệu phổ
# C là tham số điều chỉnh lỗi phân loại và biên độ
model = SVC(kernel='linear', C=1.0, random_state=42) 
model.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# --- 7. Đánh giá mô hình ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác của mô hình trên tập kiểm tra: {accuracy:.2f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- 8. Hàm phân loại một phổ mới ---
def classify_single_new_spectrum(model, new_spectrum_file_path, max_length_trained, target_wavelength=640, label_encoder=None, class_names=None):
    """
    Hàm đọc, tiền xử lý và phân loại một file phổ mới (.dpt).
    Đảm bảo các bước tiền xử lý giống hệt như dữ liệu huấn luyện.
    """
    try:
        spectrum_raw = np.loadtxt(new_spectrum_file_path, delimiter=',')
        wave = spectrum_raw[:, 0]
        intensity = spectrum_raw[:, 1]

        # Cắt phổ theo bước sóng giống như khi huấn luyện
        length_after_cut = np.searchsorted(wave, target_wavelength)
        intensity_short = intensity[:length_after_cut]

        # Chuẩn hóa dữ liệu bằng Z-score
        normalized_values = z_score_normalize(intensity_short)

        # Đảm bảo độ dài phổ mới khớp với độ dài huấn luyện bằng zero-padding
        if len(normalized_values) < max_length_trained:
            padded_spectrum = np.pad(normalized_values, (0, max_length_trained - len(normalized_values)), 'constant', constant_values=0)
        else:
            # Nếu phổ mới dài hơn max_length_trained, cắt bớt
            padded_spectrum = normalized_values[:max_length_trained]
        
        # Reshape để phù hợp với input của model.predict (1 mẫu, nhiều đặc trưng)
        new_spectrum_processed = padded_spectrum.reshape(1, -1)

        prediction = model.predict(new_spectrum_processed)
        
        if label_encoder and class_names is not None:
            # Chuyển đổi nhãn số trở lại tên lớp
            predicted_class_index = prediction[0]
            if predicted_class_index < len(class_names):
                return class_names[predicted_class_index]
            else:
                return f"Lớp không xác định (chỉ số: {predicted_class_index})"
        else:
            return prediction[0] # Trả về nhãn số nếu không có label_encoder

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file phổ mới: {new_spectrum_file_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi xử lý phổ mới '{new_spectrum_file_path}': {e}")
        return None

print("\n--- Phân loại phổ mới ---")
# Đặt đường dẫn đầy đủ đến file phổ mới bạn muốn phân loại
# Ví dụ: 'D:\Data\Fourier\Data_250502_New\pho_moi_can_phan_loai.dpt'
new_spectrum_to_classify_path = r'D:\Data\Fourier\Data_250502_New\pho_moi_can_phan_loai.dpt' 

predicted_label = classify_single_new_spectrum(model, 
                                               new_spectrum_to_classify_path, 
                                               max_spectrum_length, # max_length_trained là max_spectrum_length từ dữ liệu huấn luyện
                                               target_wavelength=640,
                                               label_encoder=le,
                                               class_names=class_names)

if predicted_label:
    print(f"Phổ '{os.path.basename(new_spectrum_to_classify_path)}' được phân loại là: {predicted_label}")
else:
    print("Không thể phân loại phổ mới.")

# --- (Tùy chọn) 9. Hiển thị một số phổ đã được tiền xử lý và tăng cường ---
plt.figure(figsize=(12, 7))
# Lấy một vài mẫu phổ từ mỗi lớp từ tập huấn luyện để hiển thị
# Lấy tối đa 3 mẫu mỗi loại để tránh biểu đồ quá dày đặc
num_samples_to_plot = min(3, len(X_train[y_train == 0]), len(X_train[y_train == 1])) 

if num_samples_to_plot > 0:
    for class_idx, class_name in enumerate(class_names):
        samples_of_class = X_train[y_train == class_idx]
        for i in range(num_samples_to_plot):
            if i < len(samples_of_class):
                plt.plot(samples_of_class[i], label=f'Phổ {class_name} Mẫu {i+1}', alpha=0.7, 
                         linestyle='-' if i % 2 == 0 else '--') # Dùng linestyle khác nhau để dễ phân biệt

plt.title('Ví dụ về Phổ đã được Tiền xử lý và Tăng cường (từ tập huấn luyện)')
plt.xlabel('Điểm dữ liệu (Sau cắt và Pad)')
plt.ylabel('Cường độ (Z-score normalized)')
plt.legend()
plt.grid(True)
plt.show()