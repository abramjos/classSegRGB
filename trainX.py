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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def save_checkpoint(state, dir, is_best=False):
  ckpt_file = os.path.join(dir, 'model.ckpt')
  torch.save(state, ckpt_file)
  if is_best:
    shutil.copyfile(
      ckpt_file, 
      os.path.join(dir, 'model_best.ckpt'))

def train_epoch(epoch, Dataloader, Criteria, Model, Optimizer, device = 'cpu'):

  classifRes = [[],[]]


  Loss = {}

  for i in Model.keys():
    Loss[i] = utils.AverageMeter()
    Model[i].train()

  for idx,datax in enumerate(Dataloader):

    # import ipdb;ipdb.set_trace()
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
        

        # if i == 'Segmentation':

    if idx%3 == 0: #Autoencoder loss 
      loss_AE = Criteria['BG'](x_hat, BGSeg)
      Loss['Decoder'].update(loss_AE.item(), BGSeg.size(0))
      loss_AE.backward()

    if idx%3 == 1: #classiffication loss 
        # if i == 'Classifier':
      for jx,jy in zip(classAct.argmax(dim=1),Pred['Classifier'].argmax(dim=1).detach().to('cpu')):
        classifRes[0].append(jx.item())
        classifRes[1].append(jy.item())

      loss_classif = Criteria['Classifier'](Pred['Classifier'],classAct.argmax(dim=1))
      Loss['Classifier'].update(loss_classif.item(), classAct.size(0))

      loss_classif.backward()

    elif idx%3 == 2: #Segmentation loss 
      loss_segment = 0
      for pred,act in zip(Pred['Segmentation'],ClassSeg):
        loss_segment += Criteria['BG'](pred,act)
      
      loss_segment/=float(classAct.sum())

      Loss['Segmentation'].update(loss_segment.item(), classAct.size(0))
      loss_segment.backward()
    
    for i in Optimizer.keys():
      Optimizer[i].step()

  return({k:loss.avg for k,loss in Loss.items()}, classifRes)

def test_epoch(epoch, Dataloader, Criteria, Model, Optimizer, device = 'cpu'):
  
  classifRes = [[],[]]


  LossEval = {}

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
    LossEval['Decoder'].update(loss_AE.item(), BGSeg.size(0))

    Pred = {'Classifier':0,'Segmentation':0}
    InNxt = {'Classifier':x_hatC,'Segmentation':x_hatS}


    for i in Model.keys():

      if i in ['Classifier','Segmentation']:
        if Model[i].encoder_only:
          Pred[i] = Model[i](InNxt[i])
        else:
          Pred[i] = Model[i](Z)
        
        if i == 'Classifier':
          for jx,jy in zip(classAct.argmax(dim=1),Pred['Classifier'].argmax(dim=1).detach().to('cpu')):
            classifRes[0].append(jx.item())
            classifRes[1].append(jy.item())

          loss_classif = Criteria['Classifier'](Pred['Classifier'],classAct.argmax(dim=1))
          LossEval['Classifier'].update(loss_classif.item(), BGSeg.size(0))


        if i == 'Segmentation':
          loss_segment = 0
          for pred,act in zip(Pred['Segmentation'],ClassSeg):
            loss_segment += Criteria['BG'](pred,act)
          
          loss_segment/=float(classAct.sum())

          LossEval['Segmentation'].update(loss_segment.item(), BGSeg.size(0))

  return({k:loss.avg for k,loss in LossEval.items()}, classifRes)

    
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
        data[i] = [item[i] for item in batch]
        # if i != 'class':
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
  
  # if args.ckpt_file:
  #   print ('loading checkpoint file ... ')
  #   if args.device == 'cpu':
  #     ckpt = torch.load(args.ckpt_file, map_location=lambda storage, loc: storage)
  #   else:
  #     ckpt = torch.load(args.ckpt_file)

  #   enc.load_state_dict(ckpt['encoder'])
  #   dec.load_state_dict(ckpt['decoder'])
  #   enc_optim.load_state_dict(ckpt['encoder_optim'])
  #   dec_optim.load_state_dict(ckpt['decoder_optim'])
  #   starting_epoch = ckpt['epoch']


  trainCriteria = {'BG': torch.nn.MSELoss(), 'Classifier':torch.nn.CrossEntropyLoss(), 'Segmentation':torch.nn.MSELoss()}
  testCriteria = {'BG': torch.nn.L1Loss(), 'Classifier':torch.nn.CrossEntropyLoss(), 'Segmentation':torch.nn.MSELoss()}

  min_test_loss = 1e5

  for epoch in trange(starting_epoch, args.n_epochs, ncols=100):

    train_loss, train_classifRes = train_epoch(epoch, Dataloader = train_loader, Criteria = trainCriteria, Model = models, Optimizer = models_optim, device=args.device)
    
    test_loss, test_classifRes = test_epoch(epoch, Dataloader = test_loader, Criteria = testCriteria, Model = models, Optimizer = models_optim, device=args.device)

    # import ipdb;ipdb.set_trace()
    testLossAll = test_loss['Decoder']+test_loss['Classifier']+test_loss['Segmentation']
    trainLossAll = train_loss['Decoder']+train_loss['Classifier']+train_loss['Segmentation']
    test_loss['Encoder'] = test_loss['Decoder']
     
    for k in models_optim.keys():
      models_optim_scheduler[k].step(test_loss[k])

    # log_value('train_loss', train_loss+train_classif_loss, epoch)
    # log_value('test_loss', test_loss+test_classif_loss, epoch)

    enc.eval(); dec.eval(); classif.eval()
    for _setname, _set in zip(['train', 'test'], [train_set, test_set]):
      n_compare = 27
      inds = np.random.choice(len(_set), n_compare)
      x_comb = []
      classifRes = []
      for i in inds:
        import ipdb;ipdb.set_trace()
        ogImage = transforms.ToPILImage()(_set[i]['combinedRGB'])
        x_comb.append(_set[i]['inputImage'].unsqueeze_(0))
        d1 = ImageDraw.Draw(ogImage)
        d1.text((10, 50), str(_set[i]['class'])+' : '+classes[_set[i]['class']], fill =(255, 255, 255))
        ogTensor = transforms.ToTensor()(ogImage).unsqueeze_(0)
        x_comb.append(ogTensor)

        Z = enc(_set[i]['inputImage'].unsqueeze(0).to(args.device))
        out,outC = dec(Z)
        out = out.detach().to('cpu')
        
        if classif.encoder_only:
          predClass = classif(outC).argmax(dim=1).detach().to('cpu').item()
        else:
          predClass = classif(Z).argmax(dim=1).detach().to('cpu').item()

        ogImagePred = transforms.ToPILImage()(out.squeeze(0))
        d2 = ImageDraw.Draw(ogImagePred)
        d2.text((10, 50), str(_set[i]['class'])+' : '+classes[predClass], fill =(255, 255, 255))
        ogTensorPred = transforms.ToTensor()(ogImagePred).unsqueeze_(0)
        x_comb.append(ogTensorPred)
        classifRes.append([_set[i]['class'],predClass])
        # import ipdb;ipdb.set_trace()
      save_image(
        make_grid(torch.cat(x_comb, 0), nrow=int((n_compare/3))), 
        os.path.join(args.exp_dir, 'visual/real_vs_reconstructed_{}_{}.png'.format(epoch,_setname)),
        2)
    
      # import ipdb;ipdb.set_trace()
    
    trainRes = '\n{}:Train\t| accuracy -{:.3f} |\tlossRGB-{:.3f} |\tloss_classif-{:.3f} |\tloss_segment-{:.3f}'.format(epoch,accuracy_score(y_true=train_classifRes[0], y_pred=train_classifRes[1]), train_loss['Decoder'], train_loss['Classifier'], train_loss['Segmentation'])
    testRes = '{}:Test\t| accuracy -{:.3f} |\tlossRGB-{:.3f} |\tloss_classif-{:.3f} |\tloss_segment-{:.3f}\n'.format(epoch,accuracy_score(y_true=test_classifRes[0], y_pred=test_classifRes[1]),  train_loss['Decoder'], train_loss['Classifier'], train_loss['Segmentation'])
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
      test_loss < min_test_loss)

    if test_loss < min_test_loss:
      min_test_loss = test_loss

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
