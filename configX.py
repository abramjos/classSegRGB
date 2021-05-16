import torch
import dataset
# from encoderX import celebA_Encoder
# from decoderX import celebA_Decoder
from classifierX import Classifier,Segment,celebA_Encoder,celebA_Decoder
import torchvision
import torchvision.transforms as transforms

from dataset import Simpload
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class config():
  def __init__(self,args):
    self.args = args

  def load_dataset(self):

    if self.args.dataset == 'celebA':
      train_transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
      test_transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()])

      celeba = dataset.celebA(
        self.args.data_dir, self.args.red_rate, self.args.test_split, self.args.validation_split)
      train_set = dataset.celebA_Subset(celeba.train_images, train_transform)
      test_set = dataset.celebA_Subset(celeba.test_images, test_transform)

    elif self.args.dataset == 'Simp':
      folder = self.args.data_dir
      imgs_dir = folder+'/img'
      coco_instance = COCO(folder+'/instances.json')


      size = (self.args.size,self.args.size)
      preprocess = transforms.Compose([ transforms.ToPILImage(), transforms.RandomVerticalFlip(0.3),transforms.RandomHorizontalFlip(0.3), transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0), transforms.ToTensor(),])
      preprocessOut = transforms.Compose([ transforms.ToPILImage(), transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0), transforms.ToTensor()])
      preprocess4D = transforms.Compose([ transforms.ToPILImage(), transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0), transforms.Resize((64,64)), transforms.ToTensor()])
      # dataset = Simpload(COCOInstance = coco_instance, imgs_dir = imgs_dir, preProcess = [preprocessOut,preprocessOut], size=size)
      
      selected_Class = ['Homer','Marge','Maggie','Lisa','Bart']
      dataset = Simpload(COCOInstance = coco_instance, imgs_dir = imgs_dir, preProcess = [preprocessOut,preprocessOut,preprocess4D], size=size, selectedCat = selected_Class)
      val_percent = self.args.test_split
      n_val = int(len(dataset) * val_percent)
      n_train = len(dataset) - n_val
      torch.manual_seed(0);
      train_set, test_set = random_split(dataset, [n_train, n_val])

    return train_set, test_set, dataset.classDict

  def load_model(self,no_classes=None, encoder_only=False, segmentor = False, Tone = 0):
    self.segmentor = segmentor


    if self.args.dataset == 'Simp':
      enc = celebA_Encoder(self.args.d_latent, self.args.device, self.args.exp_dir)
      dec = celebA_Decoder(self.args.d_latent, self.args.device, self.args.exp_dir)
      if no_classes!=None:
        # Tone = 3
        classif = Classifier(self.args.d_latent, device=self.args.device, no_classes=no_classes, encoder_only=encoder_only)
        if self.segmentor:
          Segmentator = Segment(self.args.d_latent, device=self.args.device, no_classes=no_classes, encoder_only=encoder_only, Tone = Tone)

        # asd = torch.zeros((32,256), requires_grad=True)
        # res_decoderF = decoder2(dick.cpu())

    if (self.args.device == 'cuda') and ('multi_gpu' in self.args) and (self.args.multi_gpu == True):
      print ('replicating the model on multiple gpus ... ')
      enc = torch.nn.DataParallel(enc)
      dec = torch.nn.DataParallel(dec)
      if no_classes!=None:
        if self.segmentor:
          Segmentator = torch.nn.DataParallel(Segmentator)
        classif = torch.nn.DataParallel(classif)

    if no_classes!=None:
      if self.segmentor:
        return {'Encoder':enc, 'Decoder':dec, 'Classifier':classif, 'Segmentation':Segmentator}
      else:
        return {'Encoder':enc, 'Decoder':dec, 'Classifier':classif}

    else:
      return {'Encoder':enc, 'Decoder':dec}

