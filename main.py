
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# Kiểm tra xem GPU có khả dụng không
print("CUDA Available:", torch.cuda.is_available())

# Nếu có, in tên GPU
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
