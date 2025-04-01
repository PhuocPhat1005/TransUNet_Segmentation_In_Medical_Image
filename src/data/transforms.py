import random
import numpy as np
import torch
from scipy.ndimage import rotate, zoom


class RandomGenerator:
  def __random_rot_flip(self, image, mask):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return image, mask
  
  def __random_rotate(self, image, mask):
    angle = np.random.randint(-20, 20)
    image = rotate(image, angle, order=0, reshape=False)
    mask = rotate(mask, angle, order=0, reshape=False)
    return image, mask

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    rand = random.random()
    if rand > 2/3:
      image, mask = self.__random_rot_flip(image, mask)
    elif rand > 1/3:
      image, mask = self.__random_rotate(image, mask)
    sample = {'image': image, 'mask': mask}
    return sample
  

class Zoomer:
  def __init__(self, output_size):
    self.output_size = output_size
  
  def __zoom(self, image, mask):
    x, y = image.shape
    if x != self.output_size or y != self.output_size:
      image = zoom(image, (self.output_size / x, self.output_size / y), order=3)
      mask = zoom(mask, (self.output_size / x, self.output_size / y), order=0)
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    mask = torch.from_numpy(mask.astype(np.float32))
    return image, mask
  
  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    image, mask = self.__zoom(image, mask)
    sample = {'image': image, 'mask': mask}
    return sample
  

if __name__ == '__main__':
  import os
  from torchvision import transforms
  from synapse import SynapseDataset

  path = os.path.dirname(os.path.abspath(__file__))

  for split in ['train', 'test']:
    split_path = os.path.join(path, f'./datasets/synapse/{split}')
    synapse = SynapseDataset(split_path, transform=transforms.Compose([
      RandomGenerator(), Zoomer(224)
    ]))

    print(f'Total {split} samples:', len(synapse))
    for i, sample in enumerate(synapse):
      print(f'Sample {i}:', sample['image'].shape, sample['mask'].shape)
      if i == 5: break