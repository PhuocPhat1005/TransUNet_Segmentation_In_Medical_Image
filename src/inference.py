import cv2
import numpy as np
import torch

from model.manager import ModelManager


class Inference:
  def __init__(self, args):
    self.args = args
    self.model_manager = ModelManager(args)
    self.model_manager.load_model()

  def __read_and_preprocess(self):
    image = cv2.imread(self.args.image_path)
    image_torch = cv2.resize(image, (self.args.image_dim, self.args.image_dim))
    image_torch = image_torch / 255.
    image_torch = image_torch.transpose((2, 0, 1))
    image_torch = np.expand_dims(image_torch, axis=0)
    image_torch = torch.from_numpy(image_torch.astype('float32')).to(self.device)
    return image, image_torch
  
  def __save(self, mask):
    cv2.imshow("Mask", mask)

  def __threshold(self, mask, thresh=0.5):
    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0
    return mask

  def infer(self):
    image, image_torch = self.__read_and_preprocess(self)
    with torch.no_grad():
      pred_mask = self.transunet.model(image_torch)
      pred_mask = torch.sigmoid(pred_mask)
      pred_mask = pred_mask.detach().cpu().numpy().transpose((0, 2, 3, 1))
    
    orig_h, orig_w = image.shape[:2]
    pred_mask = cv2.resize(pred_mask[0, ...], (orig_w, orig_h))
    pred_mask = self.__threshold(pred_mask, thresh=self.args.infer_threshold)
    pred_mask *= 255

    if self.args.merge_infer:
      pred_mask = cv2.bitwise_and(image, image, mask=pred_mask.astype('uint8'))
    if self.args.save_infer:
      self.__save(pred_mask)
    return pred_mask
