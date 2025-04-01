import argparse # dùng để phân tích và xử lý các tham số dòng lệnh khi chạy script
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn # quản lý các cấu hình của cuDNN -> tăng tốc độ huấn luyện trên GPU
from networks.vit_seg_modeling import VisionTransformer as ViT_seg # import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg # import CONFIGS
from trainer import trainer_synapse # import hàm huấn luyện cho tập dữ liệu Synapse

# Khởi tạo parser để nhận các tham số dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/synapse/processed/train', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='../data/synapse/list', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args() # lệnh này sẽ đọc và phân tích các tham số dòng lệnh -> lưu kết quả vào biến args


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True # cho phép cuDNN tìm kiếm thuật toán tối ưu dựa trên dữ liệu input
        cudnn.deterministic = False # không đảm bảo tính định hướng kết quả
    else:
        cudnn.benchmark = False # không tối ưu bằng benchmark
        cudnn.deterministic = True # đảm bảo các phép tính cho kết quả nhất quán khi chạy

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'synapse': {
            'root_path': '../data/synapse/processed/train',
            'list_dir': '../data/synapse/list',
            'num_classes': 9,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    # Kiểm tra xem thư mục snapshot_path đã tồn tại hay chưa. Nếu chưa, sử dụng os.makedirs để tạo thư mục đó.
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # Cấu hình mô hình ViT cho phân đoạn
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    # điều chỉnh cấu hình cho mô hình ResNet50 + ViT-B/16
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # Khởi tạo mô hình và chuyển sang GPU
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # load trọng số đã được huấn luyện trước cho mô hình
    net.load_from(weights=np.load(config_vit.pretrained_path))

    # chọn hàm huấn luyện tương ứng với dataset
    trainer = {'synapse': trainer_synapse,}
    # gọi hàm huấn luyện với các tham số đã được cấu hình
    trainer[dataset_name](args, net, snapshot_path)