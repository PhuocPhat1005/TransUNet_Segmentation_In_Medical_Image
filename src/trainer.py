import argparse # thư viện để phân tích tham số dòng lệnh
import logging # thư viện để ghi log
import os
import random
import sys # thư viện thao tác với các tham số hệ thống
import time # thư viện làm việc với thời gian
import numpy as np # thư viện xử lý mảng số học
import torch # thư viện chính cho deep learning
import torch.nn as nn # module chứa các lớp mạng noron
import torch.optim as optim # các bộ tối ứu hóa
from tensorboardX import SummaryWriter # ghi log vào TensorBoard
from torch.nn.modules.loss import CrossEntropyLoss # hàm mất mát CrossEntropyLoss
from torch.utils.data import DataLoader # DataLoader để tải dữ liệu theo batch
from tqdm import tqdm # thư viện để tạo thanh tiến trình
from utils import DiceLoss # Import hàm mất mát DiceLoss từ tệp utils
from torchvision import transforms # Module để áp dụng phép biến đổi lên hình ảnh

def trainer_synapse(args, model, snapshot_path):
    from datasets.synapse_dataset import SynapseDataset, SynapseAugmentor

    # thiết lập logging để ghi log vào file log.txt và in ra màn hình
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)) # ghi log các tham số huấn luyện

    base_lr = args.base_lr # lấy learning rate từ tham số truyền vào
    num_classes = args.num_classes # số lớp cần phân loại
    batch_size = args.batch_size * args.n_gpu # kích thước batch, nhân với số GPUs nếu dùng nhiều GPUs

    # max_iterations = args.max_iterations
    # tạo dataset cho tập huấn luyện
    db_train = SynapseDataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [SynapseAugmentor(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # hàm khởi tạo worker để đảm bảo mỗi worker có seed ngẫu nhiên khác nhau
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # tạo DataLoader để tải dữ liệu theo batch
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # nếu sử dụng nhiều GPUs, bọc mô hình trong DataParallel
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train() # Đặt mô hình ở chế độ huấn luyện

    ce_loss = CrossEntropyLoss() # Hàm mất mát CrossEntropy cho phân loại đa lớp
    dice_loss = DiceLoss(num_classes) # Hàm mất mát Dice để đo độ trùng khớp giữa nhãn và dự đoán
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # Khởi tạo optimizer SGD với momentum và weight decay

    writer = SummaryWriter(snapshot_path + '/log') # ghi log vào TensorBoard
    iter_num = 0 # biến đếm số lần lặp
    max_epoch = args.max_epochs # tổng số epoch huấn luyện
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1 # số lần lặp tối đa
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0 # biến lưu trữ hiệu suất tốt nhất
    iterator = tqdm(range(max_epoch), ncols=70) # thanh tiến trình cho epoch

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader): # lặp qua từng batch trong tập train
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'] # lấy hình ảnh và nhãn từ batch
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda() # chuyển dữ liệu sang GPU
            outputs = model(image_batch) # dự đoán kết quả từ mô hình

            loss_ce = ce_loss(outputs, label_batch[:].long()) # tính cross entropy loss
            loss_dice = dice_loss(outputs, label_batch, softmax=True) # tính dice loss
            loss = 0.5 * loss_ce + 0.5 * loss_dice # tổng hợp hai hàm mất mát với trọng số bằng nhau

            optimizer.zero_grad() # đặt lại gradient về 0 trước khi lan truyền ngược
            loss.backward() # lan truyền ngược để tính gradient
            optimizer.step() # cập nhật trọng số của mô hình

            # giảm learning rate theo số lần lặp
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1 # tăng biến đếm số lần lặp
            writer.add_scalar('info/lr', lr_, iter_num) # ghi giá trị learning rate vào TensorBoard
            writer.add_scalar('info/total_loss', loss, iter_num) # ghi tổng loss vào TensorBoard
            writer.add_scalar('info/loss_ce', loss_ce, iter_num) # ghi cross entropy loss vào TensorBoard

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # Cứ mỗi 20 iteration, ghi ảnh vào TensorBoard
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :] # Chọn một ảnh từ batch
                image = (image - image.min()) / (image.max() - image.min()) # chuẩn hóa ảnh
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6) # chu kỳ lưu model
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()  # Đóng TensorBoard writer
    return "Training Finished!" # Trả về thông báo hoàn thành huấn luyện