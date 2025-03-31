import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SynapseDataset


def visualize_train_sample(dataset):
    """
    Lấy một sample từ dataset training và hiển thị ảnh cùng nhãn.

    Ảnh được lưu dưới dạng slice 2D (định dạng .npz) với shape [1, H, W].
    """
    sample = dataset[0]  # Lấy sample đầu tiên
    # Lấy ảnh và nhãn từ sample
    image = sample['image'].squeeze()  # Chuyển [1, H, W] thành [H, W]
    label = sample['label']
    case_name = sample['case_name']

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Train Image - {case_name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap="jet")
    plt.title("Train Label")
    plt.axis("off")

    plt.show()


def visualize_test_sample(dataset):
    """
    Lấy một sample từ dataset testing và hiển thị một slice trung tâm của volume.

    Ảnh test được lưu dưới dạng volume 3D (.npy.h5) với shape [H, W, D].
    Slice trung tâm (theo chiều thứ 3) sẽ được hiển thị.
    """
    sample = dataset[0]  # Lấy sample đầu tiên
    image_volume = sample['image']  # Giả sử shape của volume là [H, W, D]
    case_name = sample['case_name']

    # Tính slice giữa theo chiều thứ 3
    mid_slice = image_volume[:, :, image_volume.shape[2] // 2]

    plt.figure(figsize=(6, 6))
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f"Test Volume Slice (middle) - {case_name}")
    plt.axis("off")
    plt.show()


def main():
    # Đường dẫn đã xử lý dữ liệu và thư mục list
    processed_train_dir = "data/acdc/processed/train"
    processed_test_dir = "data/acdc/processed/test"
    list_dir = "data/acdc/list"

    # Tạo dataset cho train và test sử dụng SynapseDataset đã có
    train_dataset = SynapseDataset(base_dir=processed_train_dir, list_dir=list_dir, split='train', transform=None)
    test_dataset = SynapseDataset(base_dir=processed_test_dir, list_dir=list_dir, split='test', transform=None)

    # Tạo DataLoader (nếu cần)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Hiển thị một sample của tập train
    print("Hiển thị sample từ tập TRAIN:")
    visualize_train_sample(train_dataset)

    # Hiển thị một sample của tập test (slice trung tâm của volume)
    print("Hiển thị sample từ tập TEST:")
    visualize_test_sample(test_dataset)

if __name__ == "__main__":
    main()
