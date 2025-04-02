import argparse
import sys
import torch

from trainer import Trainer
from inference import Inference


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'])
  parser.add_argument('--dataset_name', required='train' in sys.argv, type=str, choices=['Synapse'])
  parser.add_argument('--train_path', required='train' in sys.argv,  type=str, default=None)
  parser.add_argument('--test_path', required='train' in sys.argv, type=str, default=None)
  parser.add_argument('--save_path', required='train' in sys.argv, type=str, default=None)
  parser.add_argument('--log_path', required='train' in sys.argv, type=str, default=None)
  parser.add_argument('--pretrain_path', required='infer' in sys.argv, type=str, default=None)
  parser.add_argument('--image_path', required='infer' in sys.argv, type=str, default=None)
  parser.add_argument('--infer_save_path', type=str, default=None)

  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--patience', type=int, default=25)
  parser.add_argument('--infer_threshold', type=float, default=0.75)

  parser.add_argument('--image_dim', type=int, default=512)
  parser.add_argument('--in_channels', type=int, default=1)
  parser.add_argument('--out_channels', type=int, default=128)
  parser.add_argument('--head_num', type=int, default=4)
  parser.add_argument('--mlp_dim', type=int, default=512)
  parser.add_argument('--block_num', type=int, default=12)
  parser.add_argument('--patch_dim', type=int, default=16)
  parser.add_argument('--class_num', type=int, default=1)

  args = parser.parse_args()
  args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  if args.mode == 'train':
    trainer = Trainer(args)
    trainer.train()
  elif args.mode == 'infer':
    inference = Inference(args)
    inference.infer()
