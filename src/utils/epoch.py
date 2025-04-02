import torch
import numpy as np


class EpochCallback:
  end_training = False
  not_improved_epoch = 0

  def __init__(self, save_path, epochs, model, optimizer, monitor=None, patience=None, init_loss=np.inf):
    self.save_path = save_path
    self.epochs = epochs
    self.monitor = monitor
    self.patience = patience
    self.model = model
    self.optimizer = optimizer
    self.monitor_value = init_loss

  def __save_model(self, epoch, loss):
    torch.save({
      'epoch': epoch,
      'loss': loss,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict()
    }, self.save_path)
    print(f'Model saved to {self.save_path}')

  def epoch_end(self, epoch_num, hash):
    epoch_end_str = f'Epoch {epoch_num}/{self.epochs} - '
    for name, value in hash.items():
      epoch_end_str += f'{name}: {round(value, 4)} '
    print(epoch_end_str)

    if self.monitor is None:
      self.__save_model(epoch_num, hash[self.monitor])
    elif hash[self.monitor] < self.monitor_value:
      print(f'{self.monitor} decreased from {round(self.monitor_value, 4)} to {round(hash[self.monitor], 4)}')
      self.not_improved_epoch = 0
      self.monitor_value = hash[self.monitor]
      self.__save_model(epoch_num, hash[self.monitor])
    else:
      print(f'{self.monitor} did not decrease from {round(self.monitor_value, 4)}, model did not save!')
      self.not_improved_epoch += 1
      if self.patience is not None and self.not_improved_epoch >= self.patience:
        print("Training was stopped by callback!")
        self.end_training = True