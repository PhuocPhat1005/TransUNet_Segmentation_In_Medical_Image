import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# Đường dẫn đến thư mục chứa dữ liệu thô ACDC
RAW_DIR = os.path.join(project_root, "data", "acdc", "raw")
TRAIN_RAW_DIR = os.path.join(RAW_DIR, "training")
TEST_RAW_DIR = os.path.join(RAW_DIR, "testing")

# Đường dẫn đến thư mục chứa dữ liệu đã tiền xử lý
PROCESSED_TRAIN_DIR = os.path.join(project_root, "data", "acdc", "processed", "train")
PROCESSED_TEST_DIR_3D = os.path.join(project_root, "data", "acdc", "processed", "test", "3d")
PROCESSED_TEST_DIR_2D = os.path.join(project_root, "data", "acdc", "processed", "test", "2d")

# Đường dẫn đến file list
LIST_DIR = os.path.join(project_root, "data", "acdc", "list")

class ACDCPreprocessor:
    def __init__(self, clip_range=(-125, 275), target_shape=(224, 224)):
        self.clip_range = clip_range
        self.target_shape = target_shape

    def load_nii(self, file_path):
        nii = nib.load(file_path)
        return nii.get_fdata()

    def clip_and_normalize(self, volume):
        volume_clipped = np.clip(volume, self.clip_range[0], self.clip_range[1])
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

    def extract_slices(self, volume, label_volume=None):
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

    def process_training_data(self, image_dir, output_dir, list_file_path):
        """
        Xử lý dữ liệu huấn luyện:
         - Duyệt qua các bệnh nhân trong thư mục training (ví dụ: data/acdc/raw/training/patientXXX/).
         - Trong mỗi bệnh nhân, duyệt qua các file ảnh 3D (loại bỏ các file chứa '4d') và file nhãn tương ứng.
         - Load, clip & normalize volume.
         - Tách volume thành các slice 2D có nhãn.
         - Lưu từng slice dưới dạng file .npz và ghi danh sách tên slice vào file list_file_path.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Lấy danh sách bệnh nhân (thư mục con trong image_dir)
        # ở đây, ảnh và nhãn nằm chung trong mỗi folder patient
        patient_dirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        print("Số lượng bệnh nhân training:", len(patient_dirs))
        slice_names = []
        for patient in patient_dirs:
            patient_dir = os.path.join(image_dir, patient)
            # Lấy các file ảnh 3D không chứa _gt và '4d'
            img_files = sorted([f for f in os.listdir(patient_dir)
                                if f.endswith('.nii.gz') and '_gt' not in f and '4d' not in f])
            for img_file in img_files:
                # Nếu file ảnh kết thúc bằng '.nii.gz', loại bỏ 7 ký tự cuối để lấy case_id
                if img_file.endswith('.nii.gz'):
                    case_id = img_file[:-7]
                else:
                    case_id = os.path.splitext(img_file)[0]
                # Tạo tên file nhãn bằng cách thêm '_gt.nii.gz'
                label_file = case_id + "_gt.nii.gz"
                img_path = os.path.join(patient_dir, img_file)
                label_path = os.path.join(patient_dir, label_file)
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
                    # Resize mỗi slice về target_shape
                    img_slice_resized = self.resize_slice(img_slice, self.target_shape, order=3)
                    label_slice_resized = self.resize_slice(label_slice, self.target_shape, order=0)
                    slice_name = f"{case_id}_slice_{i:03d}"
                    out_file = os.path.join(output_dir, f"{slice_name}.npz")
                    self.save_slice_npz(img_slice_resized, label_slice_resized, out_file)
                    slice_names.append(slice_name)
        with open(list_file_path, 'w') as f:
            for name in slice_names:
                f.write(name + "\n")

    def process_testing_data_3d(self, image_dir, output_dir, list_file_path):
        """
        Xử lý dữ liệu testing 3D của ACDC:
          - Duyệt qua các bệnh nhân trong thư mục testing (ví dụ: data/acdc/raw/testing/patientYYY/).
          - Trong mỗi bệnh nhân, chọn file ảnh đại diện (chỉ file 3D, loại bỏ file chứa '4d', _gt, Info, MANDATORY, …).
          - Load, clip & normalize volume và lưu nguyên volume 3D dưới dạng file HDF5 (.npy.h5).
          - Ghi tên volume (case) vào file list_file_path.
        """
        os.makedirs(output_dir, exist_ok=True)
        patient_dirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        print("Số lượng bệnh nhân testing (3D):", len(patient_dirs))
        case_names = []
        for patient in patient_dirs:
            patient_dir = os.path.join(image_dir, patient)
            # Lấy file ảnh đại diện: chọn file đầu tiên không chứa '_gt', '4d', 'Info', 'MANDATORY'
            vol_files = sorted([f for f in os.listdir(patient_dir)
                                if f.endswith('.nii.gz') and '_gt' not in f and '4d' not in f
                                and 'Info' not in f and 'MANDATORY' not in f])
            if len(vol_files) == 0:
                continue
            vol_file = vol_files[0]
            case_id = re.sub(r'\.nii\.gz$', '', vol_file)
            img_path = os.path.join(patient_dir, vol_file)
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
                f.write(name + "\n")

    def process_testing_data_2d(self, image_dir, output_dir, list_file_path):
        """
        Xử lý dữ liệu testing dạng 2D của ACDC:
        - Duyệt qua các bệnh nhân trong thư mục testing
        - Chọn file ảnh đại diện (loại bỏ các file chứa '_gt', '4d', 'Info', 'MANDATORY', ...)
        - Load, clip & normalize volume
        - Tách volume thành các slice 2D.
        - Resize mỗi slice về self.target_shape
        - Lưu từng slice dưới dạng file.npz (với nhãn mặc định là mảng zeros) và ghi tên vào list_file_path
        """
        os.makedirs(output_dir, exist_ok=True)
        # Lấy danh sách bệnh nhân (thư mục con trong image_dir)
        # ở đây, ảnh và nhãn nằm chung trong mỗi folder patient
        patient_dirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        print("Số lượng bệnh nhân testing (2D):", len(patient_dirs))
        slice_names = []
        for patient in patient_dirs:
            patient_dir = os.path.join(image_dir, patient)
            # Lấy các file ảnh 3D không chứa _gt và '4d'
            img_files = sorted([f for f in os.listdir(patient_dir)
                                if f.endswith('.nii.gz') and '_gt' not in f and '4d' not in f])
            for img_file in img_files:
                # Nếu file ảnh kết thúc bằng '.nii.gz', loại bỏ 7 ký tự cuối để lấy case_id
                if img_file.endswith('.nii.gz'):
                    case_id = img_file[:-7]
                else:
                    case_id = os.path.splitext(img_file)[0]
                # Tạo tên file nhãn bằng cách thêm '_gt.nii.gz'
                label_file = case_id + "_gt.nii.gz"
                img_path = os.path.join(patient_dir, img_file)
                label_path = os.path.join(patient_dir, label_file)
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
                    # Resize mỗi slice về target_shape
                    img_slice_resized = self.resize_slice(img_slice, self.target_shape, order=3)
                    label_slice_resized = self.resize_slice(label_slice, self.target_shape, order=0)
                    slice_name = f"{case_id}_slice_{i:03d}"
                    out_file = os.path.join(output_dir, f"{slice_name}.npz")
                    self.save_slice_npz(img_slice_resized, label_slice_resized, out_file)
                    slice_names.append(slice_name)
        with open(list_file_path, 'w') as f:
            for name in slice_names:
                f.write(name + "\n")

class ACDCAugmentor:
    """
    ACDCAugmentor áp dụng augmentation cho ảnh 2D của dataset ACDC.
    Bao gồm các phép xoay theo bội số 90° và xoay với góc ngẫu nhiên, sau đó resize về kích thước mong muốn.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple): Kích thước (height, width) của ảnh sau khi resize.
        """
        self.output_size = output_size

    @staticmethod
    def random_rot_flip(image, label):
        """
        Áp dụng xoay (theo bội số 90°) và lật ảnh ngẫu nhiên.

        Args:
            image (np.ndarray): Ảnh 2D.
            label (np.ndarray): Nhãn tương ứng.

        Returns:
            tuple: (image, label) sau khi biến đổi.
        """
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    @staticmethod
    def random_rotate(image, label):
        """
        Xoay ảnh với góc ngẫu nhiên từ -20 đến 20 độ.

        Args:
            image (np.ndarray): Ảnh 2D.
            label (np.ndarray): Nhãn tương ứng.

        Returns:
            tuple: (image, label) sau khi xoay.
        """
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    def __call__(self, sample):
        """
        Áp dụng augmentation và resize ảnh về kích thước output_size,
        sau đó chuyển đổi sang tensor của PyTorch.

        Args:
            sample (dict): Dictionary chứa 'image' và 'label' (numpy array).

        Returns:
            dict: {'image': tensor, 'label': tensor} sau augmentation và resize.
        """
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        x, y = image.shape
        if (x, y) != self.output_size:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long()}


class ACDCDataset(Dataset):
    """
    Dataset tùy chỉnh cho bộ dataset ACDC.

    - Với split "train": Đọc file .npz chứa các slice 2D đã tiền xử lý.
    - Với split "test": Đọc file HDF5 (.npy.h5) chứa volume 3D.

    Mỗi sample trả về là một dictionary gồm:
      - 'image': tensor ảnh.
      - 'label': tensor nhãn.
      - 'case_name': tên case dựa trên file list.
    """

    def __init__(self, base_dir, list_dir, split='train', transform=None):
        """
        Args:
            base_dir (str): Thư mục chứa dữ liệu đã tiền xử lý.
            list_dir (str): Thư mục chứa file danh sách (ví dụ: train.txt, test.txt).
            split (str): "train" hoặc "test".
            transform (callable, optional): Hàm augmentation (áp dụng cho tập train).
        """
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip()
        if self.split == 'train':
            path = os.path.join(self.data_dir, case_name + '.npz')
            data = np.load(path)
            image, label = data['image'], data['label']
        else:
            if os.path.exists(os.path.join(self.data_dir, case_name + '.npz')):
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

    raw_train_dir = TRAIN_RAW_DIR  # VD: data/acdc/raw/training
    raw_test_dir = TEST_RAW_DIR

    os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROCESSED_TEST_DIR_3D, exist_ok=True)
    os.makedirs(PROCESSED_TEST_DIR_2D, exist_ok=True)
    os.makedirs(LIST_DIR, exist_ok=True)

    preprocessor = ACDCPreprocessor()

    print("\n=== Xử lý dữ liệu TRAINING ACDC ===")
    preprocessor.process_training_data(
        image_dir=raw_train_dir,
        output_dir=PROCESSED_TRAIN_DIR,
        list_file_path=os.path.join(LIST_DIR, "train.txt")
    )

    print("\n=== Xử lý dữ liệu TESTING ACDC (3D) ===")
    preprocessor.process_testing_data_3d(
        image_dir=raw_test_dir,
        output_dir=PROCESSED_TEST_DIR_3D,
        list_file_path=os.path.join(LIST_DIR, "test_3d.txt")
    )

    print("\n=== Xử lý dữ liệu TESTING ACDC (2D) ===")
    preprocessor.process_testing_data_2d(
        image_dir=raw_test_dir,
        output_dir=PROCESSED_TEST_DIR_2D,
        list_file_path=os.path.join(LIST_DIR, "test_2d.txt")
    )

    print("\nTiền xử lý raw data ACDC hoàn tất!")

    # === Augmentation & Data Loading cho tập TRAIN ===
    print("\n=== Tải dữ liệu TRAIN ACDC với Augmentation ===")
    train_transform = ACDCAugmentor(output_size=(224, 224))
    test_transform = ACDCAugmentor(output_size=(224, 224))
    train_dataset = ACDCDataset(
        base_dir=PROCESSED_TRAIN_DIR,
        list_dir=LIST_DIR,
        split='train',
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    for batch in train_loader:
        print(f"Batch TRAIN - Hình ảnh: {batch['image'].shape}, Nhãn: {batch['label'].shape}")
        break

    # === Data Loading cho tập TEST ===
    print("\n=== Tải dữ liệu TEST ACDC (3D, có Augmentation) ===")
    test_dataset_3d = ACDCDataset(
        base_dir=PROCESSED_TEST_DIR_3D,
        list_dir=LIST_DIR,
        split='test_3d',
        transform=None
    )
    test_loader_3d = DataLoader(test_dataset_3d, batch_size=1, shuffle=False, num_workers=1)
    for batch in test_loader_3d:
        print(f"Case TEST (3D) - Hình ảnh: {batch['image'].shape}, Tên: {batch['case_name']}")
        break

    print("\n=== Tải dữ liệu TEST ACDC (2D, có Augmentation) ===")
    test_dataset_2d = ACDCDataset(
        base_dir=PROCESSED_TEST_DIR_2D,
        list_dir=LIST_DIR,
        split='test_2d',
        transform=test_transform
    )
    test_loader_2d = DataLoader(test_dataset_2d, batch_size=1, shuffle=False, num_workers=1)
    for batch in test_loader_2d:
        print(f"Case TEST (2D) - Hình ảnh: {batch['image'].shape}, Tên: {batch['case_name']}")
        break

    print("\nAugmentation và kiểm tra dữ liệu ACDC đã sẵn sàng!")

if __name__ == "__main__":
    main()
