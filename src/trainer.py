from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import numpy as np

from data import datasets
from data.transforms import RandomGenerator, Zoomer
from model.manager import ModelManager
from utils.epoch import EpochCallback


class Trainer:
  def __init__(self, args):
    self.args = args
    self.train_loader = self.__load_dataset(self.args.train_path, 'train')
    self.test_loader = self.__load_dataset(self.args.test_path, 'test')
    self.model_manager = ModelManager(args)
    if (self.args.pretrain_path):
      self.init_epoch, self.init_loss = self.model_manager.load_model()
    else:
      self.init_epoch, self.init_loss = 0, np.inf

  def __load_dataset(self, path, split):
    shuffle = split == 'train'
    transform = [RandomGenerator(), Zoomer(self.args.image_dim)] if split == 'train' else [Zoomer(self.args.image_dim)]
    transform = transforms.Compose(transform)
    dataset = datasets[self.args.dataset_name](path, transform)
    loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, num_workers=4)
    return loader
  
  def __loop(self, loader, step_function, t):
    total_loss = 0
    for _, data in enumerate(loader):
      image = data['image'].to(self.args.device)
      mask = data['mask'].to(self.args.device)
      loss, _ = step_function(image=image, mask=mask)
      total_loss += loss
      t.update()
    return total_loss

  def train(self):
    callback = EpochCallback(
      save_path=self.args.save_path, epochs=self.args.epochs,
      model=self.model_manager.model, optimizer=self.model_manager.optimizer,
      monitor='test_loss', patience=self.args.patience, init_loss=self.init_loss
    )

    for epoch in range(self.init_epoch, self.args.epochs):
      with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
        train_loss = self.__loop(self.train_loader, self.model_manager.train_step, t)
        test_loss = self.__loop(self.test_loader, self.model_manager.test_step, t)

      callback.epoch_end(epoch + 1, {
        'loss': train_loss / len(self.train_loader),
        'test_loss': test_loss / len(self.test_loader)
      })
      if callback.end_training: break