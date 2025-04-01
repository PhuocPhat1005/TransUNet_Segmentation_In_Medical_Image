import os
import numpy as np
from torch.utils.data import Dataset


class SynapseDataset(Dataset):
  def __init__(self, root_path, transform=None):
    self.root_path = root_path
    self.transform = transform
    self.sample_list = os.listdir(root_path)

  def __len__(self):
    return len(self.sample_list)
  
  def __getitem__(self, idx):
    slice_name = self.sample_list[idx].strip()
    data_path = os.path.join(self.root_path, slice_name)
    data = np.load(data_path)
    image, mask = data['image'], data['label']
 
    sample = {'image': image, 'mask': mask}
    if self.transform: sample = self.transform(sample)
    sample['case_name'] = self.sample_list[idx].strip()
    return sample
  

if __name__ == '__main__':
  path = os.path.dirname(os.path.abspath(__file__))

  for split in ['train', 'test']:
    split_path = os.path.join(path, f'./datasets/synapse/{split}')
    synapse = SynapseDataset(split_path)

    print(f'Total {split} samples:', len(synapse))
    for i, sample in enumerate(synapse):
      print(f'Sample {i}:', sample['image'].shape, sample['mask'])
      if i == 5: break
