# download_synapse.py
from dotenv import load_dotenv
import synapseclient
import synapseutils
import os

def download_synapse_dataset(auth_token, dataset_id, download_path="data/raw/"):
    syn = synapseclient.Synapse() # khởi tạo đối tượng synapse
    syn.login(authToken=auth_token) # đăng nhập sử dụng personal access token (pat) được cung cấp thông qua biến auth_token

    # kiểm tra th mục data đã tồn tại chưa -> nếu chưa thì khởi tạo
    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading dataset {dataset_id} to {download_path}...")

    # tải dataset từ synapse, nếu có file trùng tên thì sẽ ghi đè theo tùy chọn 'overwrite.local'
    synapseutils.syncFromSynapse(syn, dataset_id, path=download_path)
    print(f"Dataset {dataset_id} has been synced to {download_path}.")

if __name__ == "__main__":
    dataset_id = "syn3193805"
    download_path = "data/raw/"
    load_dotenv()
    auth_token = os.getenv("SYNAPSE_AUTH_TOKEN")

    if not auth_token:
        raise ValueError("Missing SYNAPSE_AUTH_TOKEN in environment variables.")

    download_synapse_dataset(auth_token, dataset_id, download_path)
