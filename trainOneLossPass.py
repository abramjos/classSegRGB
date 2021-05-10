import os
import argparse
import shutil
import numpy as np
import torch
from math import sqrt
import torch.backends.cudnn as cudnn
from configX import config as Config
import utils
from tqdm import trange
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tensorboard_logger import configure, log_value

from dataset import Simpload
from pycocotools.coco import COCO

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from PIL import ImageDraw,Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score

import torch.nn.functional as nnf
from torch import nn

Sigmoid = nn.Sigmoid()

def save_checkpoint(state, dir, is_best=False):
  ckpt_file = os.path.join(dir, 'model.ckpt')
  torch.save(state, ckpt_file)
  if is_best:
    shutil.copyfile(
    # import ipdb;ipdb.set_trace()
      ckpt_file, 
      os.path.join(dir, 'model_best.ckpt'))

def train_epoch(epoch, Dataloader, Criteria, Model, Optimizer, device = 'cpu', threshold = 0.5):

  classifRes = [[],[]]


  Loss = {}
  accuracy = utils.AverageMeter()

  for i in Model.keys():
    Loss[i] = utils.AverageMeter()
    Model[i].train()

  for idx,datax in enumerate(Dataloader):

    for i in Model.keys():
      Model[i].zero_grad()

    classAct = datax['class'].to(device)
    inputImage = datax['inputImage'].to(device)
    BGSeg = datax['combinedRGB'].to(device)
    ClassSeg = datax['classExtract2d'].to(device)

    Z = Model['Encoder'](inputImage)
    x_hat,x_hatC,x_hatS = Model['Decoder'](Z)


    Pred = {'Classifier':0,'Segmentation':0}
    InNxt = {'Classifier':x_hatC,'Segmentation':x_hatS}


    for i in Model.keys():

      if i in ['Classifier','Segmentation']:
        if Model[i].encoder_only:
          Pred[i] = Model[i](InNxt[i])
        else:
          Pred[i] = Model[i](Z)
        
    for jx,jy in zip(classAct.argmax(dim=1),Pred['Classifier'].argmax(dim=1)):
      classifRes[0].append(jx.item())
      classifRes[1].append(jy.item())

    res = np.array(Sigmoid(Pred['Classifier']).clone().detach().to('cpu')>threshold, dtype=float)
    acc = accuracy_score(y_true=np.array(classAct.detach().to('cpu')), y_pred=res)
    accuracy.update(acc)
        # if i == 'Segmentation':

    if idx%3 == 0: #Autoencoder loss 
      loss_AE = Criteria['BG'](x_hat, BGSeg)
      Loss['Decoder'].update(loss_AE.item(), classAct.size(0))
      loss_AE.backward()

    if idx%3 == 1: #classiffication loss 
        # if i == 'Classifier':
      # loss_classif = Criteria['Classifier'](Pred['Classifier'],classAct.argmax(dim=1))
      # Error| LESS STABLE | loss_classif = Criteria['Classifier'](Sigmoid(Pred['Classifier']).type(torch.FloatTensor),classAct.type(torch.FloatTensor))

      loss_classif = Criteria['Classifier'](Pred['Classifier'], classAct.float())
      Loss['Classifier'].update(loss_classif.item(), classAct.size(0))
      loss_classif.backward()


    elif idx%3 == 2: #Segmentation loss 
      loss_segment = 0
      for pred,act in zip(Pred['Segmentation'],ClassSeg):
        loss_segment += Criteria['BG'](pred,act)

      loss_segment/=float(classAct.size(0))

      Loss['Segmentation'].update(loss_segment.item(), classAct.size(0))
      loss_segment.backward()
    
    for i in Optimizer.keys():
      Optimizer[i].step()

  return({k:loss.avg for k,loss in Loss.items()}, classifRes, accuracy.avg)

def test_epoch(epoch, Dataloader, Criteria, Model, Optimizer, device = 'cpu', threshold = 0.5):
  
  classifRes = [[],[]]


  LossEval = {}
  accuracy = utils.AverageMeter()


  for i in Model.keys():
    LossEval[i] = utils.AverageMeter()
    Model[i].eval()

  # import ipdb;ipdb.set_trace()
  for idx,datax in enumerate(Dataloader):

    classAct = datax['class'].to(device)
    inputImage = datax['inputImage'].to(device)
    BGSeg = datax['combinedRGB'].to(device)
    ClassSeg = datax['classExtract2d'].to(device)



    Z = Model['Encoder'](inputImage)
    x_hat,x_hatC,x_hatS = Model['Decoder'](Z)

    loss_AE = Criteria['BG'](x_hat, BGSeg)
    LossEval['Decoder'].update(loss_AE.item(), classAct.size(0))

    Pred = {'Classifier':0,'Segmentation':0}
    InNxt = {'Classifier':x_hatC,'Segmentation':x_hatS}


    for i in Model.keys():

      if i in ['Classifier','Segmentation']:
        if Model[i].encoder_only:
          Pred[i] = Model[i](InNxt[i])
        else:
          Pred[i] = Model[i](Z)
        
        for jx,jy in zip(classAct.argmax(dim=1),Pred['Classifier'].argmax(dim=1).detach().to('cpu')):
          classifRes[0].append(jx.item())
          classifRes[1].append(jy.item())

        res = np.array(Sigmoid(Pred['Classifier']).clone().detach().to('cpu')>threshold, dtype=float)
        acc = accuracy_score(y_true=np.array(classAct.detach().to('cpu')), y_pred=res)
        accuracy.update(acc)

        if i == 'Classifier':
          # loss_classif = Criteria['Classifier'](Pred['Classifier'],classAct.argmax(dim=1))
          # Error| LESS STABLE | loss_classif = Criteria['Classifier'](Sigmoid(Pred['Classifier']).type(torch.FloatTensor),classAct.type(torch.FloatTensor))
          loss_classif = Criteria['Classifier'](Pred['Classifier'], classAct.float())
          LossEval['Classifier'].update(loss_classif.item(), classAct.size(0))
    



        if i == 'Segmentation':
          loss_segment = 0
          for pred,act in zip(Pred['Segmentation'],ClassSeg):
            loss_segment += Criteria['BG'](pred,act)
          
          loss_segment/=classAct.size(0)

          LossEval['Segmentation'].update(loss_segment.item(), classAct.size(0))

  return({k:loss.avg for k,loss in LossEval.items()}, classifRes, accuracy.avg)

    
def main(args):
  os.makedirs(args.exp_dir+'/visual',exist_ok = True)
  config = Config(args)
  train_set, test_set, classes = config.load_dataset()
  print(train_set.indices)
  print(test_set.indices)

  def collate_fn(batch):
    k = ['class', 'inputImage', 'combinedMask', 'combinedRGB', 'classExtract', 'classExtract2d']
    data = {}
    for i in k:
      if i not in data.keys():
        if i != 'class':
          data[i] = [item[i] for item in batch]
        else:
          data[i] = [item[i][1] for item in batch]
        data[i] = torch.cat(data[i], dim = 0)
    return (data)

  # import ipdb;ipdb.set_trace()
  train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
  test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)


  models = config.load_model(no_classes=train_set.dataset.noClass,encoder_only=True, segmentor =True, Tone=3)
  
  
  models_optim = {}
  models_optim_scheduler = {}
 
  for model_name,model in models.items():
    models_optim[model_name] = torch.optim.Adam(model.parameters(), 1e-3)
    models_optim_scheduler[model_name] = torch.optim.lr_scheduler.MultiStepLR(models_optim[model_name], [100, 150, 200], 0.5)
  
  starting_epoch = 0
  with open(args.logfile,'w') as f:
    f.writelines('\n\nTest\n')
  
  if args.ckpt_file:
    print ('loading checkpoint file ... ')
    if args.device == 'cpu':
      ckpt = torch.load(args.ckpt_file, map_location=lambda storage, loc: storage)
    else:
      ckpt = torch.load(args.ckpt_file)

    models['Encoder'].load_state_dict(ckpt['encoder'])
    models['Decoder'].load_state_dict(ckpt['decoder'])
    models['Classifier'].load_state_dict(ckpt['classifer'])
    models['Segmentation'].load_state_dict(ckpt['segmentor'])


    models_optim['Encoder'].load_state_dict(ckpt['encoder_optim'])
    models_optim['Decoder'].load_state_dict(ckpt['decoder_optim'])
    models_optim['Classifier'].load_state_dict(ckpt['classifer_optim'])
    models_optim['Segmentation'].load_state_dict(ckpt['segmentor_optim'])
    starting_epoch = ckpt['epoch']



  # trainCriteria = {'BG': torch.nn.MSELoss(), 'Classifier':torch.nn.CrossEntropyLoss(), 'Segmentation':torch.nn.MSELoss()}
  trainCriteria = {'BG': torch.nn.MSELoss(), 'Classifier':torch.nn.BCEWithLogitsLoss(torch.ones([5], device=args.device)), 'Segmentation':torch.nn.MSELoss()}#add bias to BCEwithLOgits
  # testCriteria = {'BG': torch.nn.L1Loss(), 'Classifier':torch.nn.CrossEntropyLoss(), 'Segmentation':torch.nn.MSELoss()}
  testCriteria = {'BG': torch.nn.L1Loss(), 'Classifier':torch.nn.BCEWithLogitsLoss(), 'Segmentation':torch.nn.MSELoss()}

  min_test_loss = 1e5

  for epoch in trange(starting_epoch, args.n_epochs, ncols=100):

    train_loss, train_classifRes, train_accuracy = train_epoch(epoch, Dataloader = train_loader, Criteria = trainCriteria, Model = models, Optimizer = models_optim, device=args.device)
    
    test_loss, test_classifRes, test_accuracy = test_epoch(epoch, Dataloader = test_loader, Criteria = testCriteria, Model = models, Optimizer = models_optim, device=args.device)

    # import ipdb;ipdb.set_trace()
    testLossAll = test_loss['Decoder']+test_loss['Classifier']+test_loss['Segmentation']
    trainLossAll = train_loss['Decoder']+train_loss['Classifier']+train_loss['Segmentation']
    test_loss['Encoder'] = test_loss['Decoder']

    for k in models_optim.keys():
      models_optim_scheduler[k].step(test_loss[k])

    # log_value('train_loss', train_loss+train_classif_loss, epoch)
    # log_value('test_loss', test_loss+test_classif_loss, epoch)

    for i in models.keys():
      models[i].eval()

    factor = 1
    reSize = (args.size*factor, args.size*factor) 

    for _setname, _set in zip(['train', 'test'], [train_set, test_set]):
      n_compare = 12
      inds = np.random.choice(len(_set), n_compare)
      AE_comb = []
      Seg_comb = []
      classifRes = []


      for i in inds:
        # import ipdb;ipdb.set_trace()
        AE_comb.append(nnf.interpolate(_set[i]['inputImage'], size=reSize, mode='bicubic', align_corners=False))
        
        ogImage = transforms.ToPILImage()(_set[i]['combinedRGB'].squeeze())
        d1 = ImageDraw.Draw(ogImage)
        d1.text((10, 50), str(_set[i]['class'][1].argmax(dim=1).item())+' : '+classes[_set[i]['class'][1].argmax(dim=1).item()], fill =(255, 255, 255))
        
        ogTensor = transforms.ToTensor()(ogImage).unsqueeze_(0)
        # AE_comb.append(ogTensor)
        AE_comb.append(nnf.interpolate(ogTensor, size=reSize, mode='bicubic', align_corners=False))

        Z = models['Encoder'](_set[i]['inputImage'].to(args.device))
        out,outC,outS = models['Decoder'](Z)
        out = out.detach().to('cpu')
        
        if models['Classifier'].encoder_only:
          predClass = models['Classifier'](outC).argmax(dim=1).detach().to('cpu').item()
        else:
          predClass = models['Classifier'](Z).argmax(dim=1).detach().to('cpu').item()

        ogImagePred = transforms.ToPILImage()(out.squeeze(0))
        d2 = ImageDraw.Draw(ogImagePred)
        d2.text((10, 50), str(_set[i]['class'][1].argmax(dim=1).item())+' : '+classes[predClass], fill =(255, 255, 255))
        ogTensorPred = transforms.ToTensor()(ogImagePred).unsqueeze_(0)
        
        # AE_comb.append(ogTensorPred)
        AE_comb.append(nnf.interpolate(ogTensorPred, size=reSize, mode='bicubic', align_corners=False))

        classifRes.append([_set[i]['class'][1].argmax(dim=1).item(),predClass])


        if 'Segmentation' in models.keys() and models['Segmentation'].encoder_only:
          SegRes = models['Segmentation'](outS).detach().to('cpu')
        else:
          SegRes = models['Segmentation'](Z).detach().to('cpu')

        if models['Segmentation'].Tone == 3:
          RGBSeg = torch.hstack([ix for ix in _set[i]['classExtract2d'].squeeze()])
          RGBOut = torch.hstack([ix for ix in SegRes.squeeze()])
          Out2d = torch.cat((RGBOut,RGBSeg), dim = 2)
        else:
          BWSeg = torch.vstack([ix for ix in _set[i]['classExtract'].squeeze()])
          BWOut = torch.vstack([ix for ix in SegRes.squeeze()])
          Out2d = torch.cat((BWOut,BWSeg), dim = 1)

        Seg_comb.append(Out2d.unsqueeze(0))




      save_image(
        make_grid(torch.cat(AE_comb, 0), pad_value= 1, nrow=int((n_compare/2))), 
        os.path.join(args.exp_dir, 'visual/real_vs_reconstructed_{}_{}.png'.format(epoch,_setname)),2)


      # import ipdb;ipdb.set_trace()
      save_image(
        make_grid(torch.cat(Seg_comb, 0), pad_value= 1, nrow=int((n_compare/2))), 
        os.path.join(args.exp_dir, 'visual/Seg_real_vs_reconstructed_{}_{}.png'.format(epoch,_setname)),2)

    
      # import ipdb;ipdb.set_trace()
    
    trainRes = '\n\n {}:Train | accuracy -{} || {:.3f} |\tlossRGB-{:.3f} |\tloss_classif-{:.3f}  |\tloss_segment-{:.3f}'.format(epoch, train_accuracy*100, accuracy_score(y_true=train_classifRes[0], y_pred=train_classifRes[1]), train_loss['Decoder'], train_loss['Classifier'], train_loss['Segmentation'])
    testRes = '{}:Test  | accuracy -{} || {:.3f} |\tlossRGB-{:.3f} |\tloss_classif-{:.3f}  |\tloss_segment-{:.3f}\n\n'.format(epoch, test_accuracy*100, accuracy_score(y_true=test_classifRes[0], y_pred=test_classifRes[1]),  train_loss['Decoder'], train_loss['Classifier'], train_loss['Segmentation'])
    
    print(trainRes,'\n',testRes)
    with open(args.logfile,'a') as f:
        f.writelines(trainRes)
        f.writelines('\n')
        f.writelines(testRes)   


    # print(classification_report(y_true=test_classifRes[0], y_pred=test_classifRes[1], target_names=[classes[i] for i in range(len(classes))]))
    print(confusion_matrix(y_true=train_classifRes[0], y_pred=train_classifRes[1]))
    save_checkpoint(
      {
        'epoch': epoch,
        'encoder': models['Encoder'].state_dict(), 
        'decoder': models['Decoder'].state_dict(),
        'classifer': models['Classifier'].state_dict(),
        'segmentor': models['Segmentation'].state_dict(),
        'encoder_optim': models_optim['Encoder'].state_dict(),
        'decoder_optim': models_optim['Decoder'].state_dict(),
        'classifer_optim': models_optim['Classifier'].state_dict(),
        'segmentor_optim': models_optim['Segmentation'].state_dict(),
        'test_loss': test_loss
      },
      args.exp_dir, 
      testLossAll < min_test_loss)

    if testLossAll < min_test_loss:
      min_test_loss = testLossAll

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='')
  parser.add_argument('--exp_dir', type=str, default='./test/')
  parser.add_argument('--ckpt_file', type=str, default='')
  parser.add_argument('--device', type=str, default='cuda')
  # parser.add_argument('--multi_gpu', action='store_true')
  parser.add_argument('--dataset', type=str, default='Simp')
  parser.add_argument('--test_split', type=float, default=0.2)
  parser.add_argument('--red_rate', type=float, default=0.0)
  parser.add_argument('--validation_split', type=float, default=0.0)
  parser.add_argument('--d_latent', type=int, default=256)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--n_epochs', type=int, default=2000)
  parser.add_argument('--size', type=int, default=256)
  parser.add_argument('--logfile', type=str, default='test.txt')
  args = parser.parse_args()

  if args.device == 'cuda' and torch.cuda.is_available():
    from subprocess import call
    print ('available gpus:')
    call(["nvidia-smi", 
           "--format=csv", 
           "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    cudnn.benchmark = True
  else:
    args.device = 'cpu'
  utils.prepare_directory(args.exp_dir)
  utils.write_logs(args)
  configure(args.exp_dir)
  main(args)