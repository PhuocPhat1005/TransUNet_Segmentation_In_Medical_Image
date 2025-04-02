import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn.modules.loss import CrossEntropyLoss

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

    self.dice_loss = DiceLoss(self.args.class_num)
    self.ce_loss = CrossEntropyLoss()
    self.optimizer = SGD(
      self.model.parameters(), lr=args.learning_rate,
      momentum=args.momentum, weight_decay=args.weight_decay
    )

  def load_model(self):
    ckpt = torch.load(self.args.pretrain_path, map_location=torch.device(self.args.device), weights_only=True)
    self.model.load_state_dict(ckpt['model_state_dict'])
    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f'Checkpoint is loaded - epoc: {ckpt["epoch"]} loss: {ckpt["loss"]}')
    return ckpt['epoch'], ckpt['loss']

  def train_step(self, image, mask):
    self.model.train()
    self.optimizer.zero_grad()
    pred_mask = self.model(image)
    loss_ce = self.ce_loss(pred_mask, mask[:].long())
    loss_dice = self.dice_loss(pred_mask, mask, softmax=True)
    loss = (loss_ce + loss_dice) / 2
    loss.backward()
    self.optimizer.step()
    return loss.item(), pred_mask

  def test_step(self, image, mask):
    self.model.eval()
    pred_mask = self.model(image)
    loss_ce = self.ce_loss(pred_mask, mask[:].long())
    loss_dice = self.dice_loss(pred_mask, mask, softmax=True)
    loss = (loss_ce + loss_dice) / 2
    return loss.item(), pred_mask