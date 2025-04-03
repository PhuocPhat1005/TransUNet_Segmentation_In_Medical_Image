import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import random
import h5py
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"Project root directory: {project_root}")

RAW_DIR = os.path.join(project_root, "data", "synapse", "raw")
TRAIN_IMG_DIR = os.path.join(RAW_DIR, "averaged-training-images")
TRAIN_LABEL_DIR = os.path.join(RAW_DIR, "averaged-training-labels")
TEST_IMG_DIR = os.path.join(RAW_DIR, "averaged-testing-images")

PROCESSED_TRAIN_DIR = os.path.join(project_root, "data", "synapse", "processed", "train")
PROCESSED_TEST_DIR = os.path.join(project_root, "data", "synapse", "processed", "test")
LIST_DIR = os.path.join(project_root, "data", "synapse", "list")

class SynapsePreprocessor:
    """
    Class Preprocessor cung cấp các phương thức để xử lý raw data cho bộ dataset synapse.

    Các chức năng chính bao gồm:
    - Tải dữ liệu NIfIT (.nii.gz) sử dụng nibabel
    - Clip giá trị pixel trong khoảng cho trước và normalize về [0, 1]
    - Tách volume 3D thành các slice 2D (có thể kèm nhãn nếu có).
    """

    def __init__(self, clip_range=(-125, 275), target_shape=(224, 224)):
        self.clip_range = clip_range
        self.target_shape = target_shape

    #  Hàm load file NIfIT (.nii.gz) dùng nibabel
    def load_nii(self, file_path):
        nii = nib.load(file_path)
        return nii.get_fdata()

    # Hàm clip giá trị pixel và normalize volume 3D
    def clip_and_normalize(self, volume):
        # CLip các giá trị pixel trong khoảng [-125, 275]
        volume_clipped = np.clip(volume, self.clip_range[0], self.clip_range[1])
        # Normalize về khoảng [0, 1] bằng công thức min-max normalization
        volume_norm = (volume_clipped - self.clip_range[0]) / (self.clip_range[1] - self.clip_range[0])
        return volume_norm

    def resize_slice(self, slice_img, target_shape, order=3):
        """
        Resize một slice ảnh về kích thước target_shape.

        Args:
            slice_img (np.ndarray): Ảnh 2D.
            target_shape (tuple): (height, width) mong muốn.
            order (int): Thứ tự nội suy (order=3 cho ảnh, order=0 cho nhãn).
        Returns:
            np.ndarray: Ảnh đã được resize.
        """
        current_shape = slice_img.shape
        zoom_factors = (target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])
        return zoom(slice_img, zoom_factors, order=order)

    # Hàm tách các slice 2D từ volume 3D
    def extract_slices(self, volume, label_volume = None):
        """
        Tách volume 3D thành các slice 2D.
        Nếu label_volume được cung cấp, chỉ trả về các slice có chứa nhãn (khác toàn 0).
        Args:
            volume (np.ndarray): Volume ảnh 3D.
            label_volume (np.ndarray, optional): Volume nhãn 3D. Mặc định None.
        Returns:
            list: Danh sách các slice. Nếu label_volume có giá trị, trả về list các tuple (img_slice, label_slice), ngược lại trả về list các img_slice.
        """
        slices = []
        num_slices = volume.shape[2]
        for i in range(num_slices):
            img_slice = volume[:, :, i]
            if label_volume is not None:
                label_slice = label_volume[:, :, i]
                if np.sum(label_slice) == 0:
                    continue
                slices.append((img_slice, label_slice))
            else:
                slices.append(img_slice)
        return slices

    def save_slice_npz(self, image_slice, label_slice, out_path):
        np.savez_compressed(out_path, image=image_slice, label=label_slice)

    def save_volume_h5(self, volume, out_path):
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("image", data=volume)

    def process_training_data(self, image_dir, label_dir, output_dir, list_file_path):
        os.makedirs(output_dir, exist_ok=True)
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
        print("Số lượng file training image:", len(image_files))
        slice_names = []

        for img_file in image_files:
            case_id = img_file.split("_avg")[0]
            label_file = case_id + "_avg_seg.nii.gz"

            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            try:
                volume = self.load_nii(img_path)
                label_volume = self.load_nii(label_path)
            except Exception as e:
                print(f"Lỗi khi load file cho {case_id}: {e}")
                continue

            volume_norm = self.clip_and_normalize(volume)
            slices = self.extract_slices(volume_norm, label_volume)

            print(f"{case_id}: Tách được {len(slices)} slice có nhãn")

            for i, (img_slice, label_slice) in enumerate(slices):
                # Resize mỗi slice vể target_shape
                img_slice_resized = self.resize_slice(img_slice, self.target_shape, order=3)
                label_slice_resized = self.resize_slice(label_slice, self.target_shape, order=0)
                slice_name = f"{case_id}_slice_{i:03d}"
                out_file = os.path.join(output_dir, f"{slice_name}.npz")
                self.save_slice_npz(img_slice_resized, label_slice_resized, out_file)
                slice_names.append(slice_name)

        with open(list_file_path, 'w') as f:
            for name in slice_names:
                f.write(name + '\n')

    def process_testing_data(self, image_dir, output_dir, list_file_path):
        os.makedirs(output_dir, exist_ok=True)
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
        print("Số lượng file testing image:", len(image_files))
        case_names = []

        for img_file in image_files:
            case_id = img_file.split("_avg")[0]
            img_path = os.path.join(image_dir, img_file)

            try:
                volume = self.load_nii(img_path)
            except Exception as e:
                print(f"Lỗi khi load file testing cho {case_id}: {e}")
                continue
            volume_norm = self.clip_and_normalize(volume)
            # Resize từng slice 2D của volume về self.target_shape
            resized_slices = []
            num_slices = volume_norm.shape[2]
            for i in range(num_slices):
                slice_img = volume_norm[:, :, i]
                slice_img_resized = self.resize_slice(slice_img, self.target_shape, order=3)
                resized_slices.append(slice_img_resized)
            resized_volume = np.stack(resized_slices, axis=2)
            out_file = os.path.join(output_dir, f"{case_id}.npy.h5")
            self.save_volume_h5(resized_volume, out_file)
            print(f"Lưu testing volume cho {case_id} với shape {resized_volume.shape} vào {out_file}")
            case_names.append(case_id)

        with open(list_file_path, 'w') as f:
            for name in case_names:
                f.write(name + '\n')

class SynapseAugmentor:
    # Hàm biến đổi: xoay ảnh 90 độ, và lật ảnh ngẫu nhiên
    @staticmethod
    def random_rot_flip(image, label):
        """
        Áp dụng phép xoay (theo bội số của 90 độ) và lật ảnh ngẫu nhiên
        Args:
            image (np.ndarray): Ảnh 2D
            label (np.ndarray): Nhãn tương ứng

        Returns:
            tuple: (image, label) sau khi biến đổi
        """
        k = np.random.randint(0, 4) # xoay theo bội số của 90 độ
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2) # Chọn trục lật ngẫu nhiên (0 hoặc 1)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    # Hàm biến đổi: xoay ảnh với góc ngẫu nhiên từ -20 đến 20 độ
    @staticmethod
    def random_rotate(image, label):
        """
        Xoay ảnh với góc ngẫu nhiên từ -20 đến 20 độ

        Args:
            image (np.ndarray): Ảnh 2D.
            label (np.ndarray): Nhãn tương ứng.

        Returns:
            tuple: (image, label) sau khi xoay.
        """
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order = 0, reshape = False)
        label = ndimage.rotate(label, angle, order = 0, reshape = False)
        return image, label

    def __init__(self, output_size):
        """
        Khởi tạo Augmentor với kích thước đầu ra mong muốn

        Args:
            output_size (tuple): Kích thước (height, width) của ảnh sau khi resize.
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        Áp dụng augmentation và resize ảnh về kích thước output_size
        Sau đó chuyển đổi kết quả sang tensor của PyTorch
        Args:
            sample (dict): Dictionary chứa 'image' và 'label' (dạng numpy array).

        Returns:
            dict: Sample đã được augmentation, resize và chuyển sang tensor.
                  Cấu trúc: {'image': tensor, 'label': tensor}
        """
        image, label = sample['image'], sample['label']

        # Áp dụng augmentation ngẫu nhiên
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)

        # Resize nếu cần thiết
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Chuyển đổi sang tensor, thêm chiều channel (unsqueeze(0) cho ảnh)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        return {'image': image, 'label': label.long()}


class SynapseDataset(Dataset):
    """
    Dataset tùy chỉnh cho bộ dữ liệu synapse.

    Hỗ trợ đọc file:
      - Với split "train": đọc file .npz chứa các slice 2D đã tiền xử lý.
      - Với split "test": đọc file HDF5 (.npy.h5) chứa volume 3D.

    Mỗi sample trả về là một dictionary bao gồm:
      - 'image': tensor ảnh.
      - 'label': tensor nhãn.
      - 'case_name': tên case được lấy từ danh sách file.
    """

    def __init__(self, base_dir, list_dir, split='train', transform=None):
        """
        Khởi tạo dataset.

        Args:
            base_dir (str): Thư mục chứa dữ liệu đã tiền xử lý.
            list_dir (str): Thư mục chứa file danh sách case (vd: train.txt, test.txt).
            split (str): Chế độ dữ liệu "train" hoặc "test".
            transform (callable, optional): Hàm biến đổi (augmentation) sẽ được áp dụng lên sample.
        """
        self.transform = transform
        self.split = split
        # Đọc danh sách case từ file text
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        """Trả về số lượng sample trong dataset."""
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip()
        if self.split == 'train':
            path = os.path.join(self.data_dir, case_name + '.npz')
            data = np.load(path)
            image, label = data['image'], data['label']
        else:
            path = os.path.join(self.data_dir, f"{case_name}.npy.h5")
            data = h5py.File(path, 'r')
            image = data['image'][:]
            label = data.get('label', np.zeros_like(image))[:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = case_name
        return sample

def main():
    print("Raw Train Image Dir:", TRAIN_IMG_DIR)
    print("Raw Train Label Dir:", TRAIN_LABEL_DIR)
    print("Raw Test Dir:", TEST_IMG_DIR)
    print("Processed Train Dir:", PROCESSED_TRAIN_DIR)
    print("Processed Test Dir:", PROCESSED_TEST_DIR)
    print("List Dir:", LIST_DIR)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)
    os.makedirs(LIST_DIR, exist_ok=True)

    preprocessor = SynapsePreprocessor()

    print("\n=== Xử lý dữ liệu TRAINING ===")
    preprocessor.process_training_data(
        image_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
        output_dir=PROCESSED_TRAIN_DIR,
        list_file_path=os.path.join(LIST_DIR, "train.txt")
    )

    print("\n=== Xử lý dữ liệu TESTING ===")
    preprocessor.process_testing_data(
        image_dir=TEST_IMG_DIR,
        output_dir=PROCESSED_TEST_DIR,
        list_file_path=os.path.join(LIST_DIR, "test.txt")
    )

    print("\nTiền xử lý raw data hoàn tất!")

    # === Augmentation & Testing sau preprocessing ===
    print("\n=== Augmentation & Tải dữ liệu TRAINING ===")
    train_transform = SynapseAugmentor(output_size=(224, 224))
    train_dataset = SynapseDataset(
        base_dir=PROCESSED_TRAIN_DIR,
        list_dir=LIST_DIR,
        split='train',
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Kiểm tra một batch training
    for batch in train_loader:
        print(f"Batch TRAIN - Hình ảnh: {batch['image'].shape}, Nhãn: {batch['label'].shape}")
        break

    print("\n=== Tải dữ liệu TESTING ===")
    test_dataset = SynapseDataset(
        base_dir=PROCESSED_TEST_DIR,
        list_dir=LIST_DIR,
        split='test',
        transform=None  # Không cần augment tập test
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Kiểm tra một volume testing
    for batch in test_loader:
        print(f"Case TEST - Hình ảnh: {batch['image'].shape}, Tên: {batch['case_name']}")
        break

    print("\nAugmentation và kiểm tra dữ liệu đã sẵn sàng!")

if __name__ == "__main__":
    main()