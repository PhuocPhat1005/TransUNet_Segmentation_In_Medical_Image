import torch
import torch.nn as nn
from torch.optim import SGD

from .criterion import DiceLoss
from .transunet import TransUNet


class ModelManager:
  def __init__(self, args):
    self.args = args
    self.model = TransUNet(
      args.image_dim, args.in_channels, args.out_channels, args.head_num,
      args.mlp_dim, args.block_num, args.patch_dim, args.class_num
    )
    self.model = nn.DataParallel(self.model)
    self.model.to(args.device)

    self.criterion = DiceLoss(self.args.class_num)
    self.optimizer = SGD(
      self.model.parameters(), lr=args.learning_rate,
      momentum=args.momentum, weight_decay=args.weight_decay
    )

  def load_model(self):
    ckpt = torch.load(self.args.pretrain_path)
    self.model.load_state_dict(ckpt['model_state_dict'])
    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['loss']

  def train_step(self, image, mask):
    self.model.train()
    self.optimizer.zero_grad()
    pred_mask = self.model(image)
    loss = self.criterion(pred_mask, mask)
    loss.backward()
    self.optimizer.step()
    return loss.item(), pred_mask

  def test_step(self, image, mask):
    self.model.eval()
    pred_mask = self.model(image)
    loss = self.criterion(pred_mask, mask)
    return loss.item(), pred_mask