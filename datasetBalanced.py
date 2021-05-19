import json
import random
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from pycocotools.coco import COCO
from coco2voc_aux import *
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from pycocotools import mask as maskUtils
import numpy as np
from torch.utils.data import DataLoader, random_split
import cv2

# from dataset import Simpload
from pycocotools.coco import COCO

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annsToMask(anns, h, w, size):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.
    :return: a list of binary masks (each a numpy 2D array) of all the annotations in anns
    """
    masks = []
    anns = sorted(anns, key=lambda x: x['area'])  # Smaller items first, so they are not covered by overlapping segs
    for ann in anns:
        rle = annToRLE(ann, h, w)
        m = maskUtils.decode(rle)
        mPIL = Image.fromarray(m,mode='L')
        masks.append(np.array(mPIL.resize(size)))
    return np.array(masks), anns

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

        

class Simpload(Dataset):
    def __init__(self, COCOInstance, imgs_dir, preProcess=None,  size=None, scale=1, mask_suffix='', maskForegroundFlag = True, selectedCat=None):
        self.imgs_dir = imgs_dir
        self.COCOInstance = COCOInstance
        self.scale = scale

        self.mask_suffix = mask_suffix
        self.maskCombine = maskForegroundFlag
        
        self.preProcess = preProcess
        self.size = size

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.coco_imgs = COCOInstance.imgs        
        self.Class = COCOInstance.cats 
        self.noClass = len(self.Class)
        self.classDict =  {i-1:v['name'] for i,v in self.Class.items()}
        self.noFilters = self.noClass#segmentation mask

        self.selectedCat = selectedCat
        # import ipdb;ipdb.set_trace() 
        self.ids = [i for i in self.coco_imgs.keys()]

        if self.selectedCat!=None:
            self.noSelected = len(self.selectedCat)
            self.selectedCatDict = {v:k for k,v in enumerate(self.selectedCat)}
        else:
            self.noSelected = self.noClass
            self.selectedCatDict = {v:k for k,v in self.classDict.items()}

        self.dict = self._getDistribution__()
        self.mult =  3
        counts = [len(v) for v in self.Dict.values()]
        counts.sort()

        self.avgInst = int(np.mean(counts[:3]))

        # logging.info('Creating dataset with {} examples with {} classes'.format(len(self.ids),len(classesT)))
    def __len__(self):
        return int(self.mult*self.avgInst*5)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
        
    def getLevelMasks(self, masks, anns_ids, image = None, no_levels=3, GaussianFilter = True, RGB = True):
        # 
        # RGB : RGB in o/p mask if image is also given
        # no of level for available features(based on convolution erode RGB/gray)
        # 
        if type(image) == np.ndarray:
            if RGB:
                multiMasks = np.zeros((self.noFilters+3, no_levels, *(self.size))).astype('int16')
            else:
                multiMasks = np.zeros((self.noFilters+1, no_levels, *(self.size))).astype('int16')
        else:
            multiMasks = np.zeros((self.noFilters, no_levels, *(self.size))).astype('int16')
        

        # kernel = np.ones((6, 6), 'uint8')
        gkernel = cv2.getGaussianKernel(ksize=(4),sigma=2)
        kernel = np.matmul(gkernel,gkernel.transpose(1,0))
        kernel = kernel/kernel.max()

        try:
            for _id,(mask,idx) in enumerate(zip(masks,anns_ids)):
                
                for iterVal in range(no_levels):

                    if GaussianFilter:
                        if type(image) == np.ndarray  and _id == 0:
                            if RGB:
                                erode_image = cv2.erode(image, kernel, cv2.BORDER_REFLECT, iterations=iterVal)
                                multiMasks[self.noFilters:, iterVal, :, :] =  erode_image.reshape((3,*(self.size))) #np.expand_dims(erode_image, axis=0).reshape((3,1,*(self.size)))
                            else:
                                erode_image = cv2.erode(cv2.CvtColor(image,cv2.COLORBGR2GRAY), kernel, cv2.BORDER_REFLECT, iterations=iterVal)
                                multiMasks[self.noFilters:, iterVal, :, :] = erode_image

                        erode_mask = cv2.erode(mask, kernel, cv2.BORDER_REFLECT, iterations=iterVal)
                        multiMasks[idx-1, iterVal, :, :] += erode_mask
        except:
            import ipdb;ipdb.set_trace()
 
        return multiMasks


    def getCombinedMask(self, masks, anns_ids, iterVals=None,  image=None, RGB=True):
        # 
        # RGB : RGB in o/p mask if image is also given
        # no of level for available features(based on convolution erode RGB/gray)
        # 


        if type(masks) == np.ndarray:
            Masks = masks[self.noFilters:,:,:,:]
            # Masks =
            MasksSelected = [] 
            for iterVal in iterVals:
                if type(iterVals) == (type([]) or np.ndarray) and iterVal in iterVals:
                    MasksSelected.append(Masks[:,iterVal,:,:].reshape(*(self.size),Masks.shape[0]))
                else:
                    MasksSelected.append(Masks[:,iterVal,:,:].reshape(*(self.size),Masks.shape[0]))

            MasksSelected = np.array(MasksSelected)
            Masksavg = np.average(MasksSelected,axis=0).reshape((*(self.size),3)).astype(np.uint8)

        # import ipdb;ipdb.set_trace()
        classExtract = np.zeros((self.noSelected,*(self.size)))
        classExtract2d = np.zeros((self.noSelected,*(self.size),3))
        combinedMask = np.zeros(self.size)
        combinedRGB = np.zeros((*(self.size),3))

        RGBoGray = Masksavg.astype(np.uint8)
        
        for idx in anns_ids:
            if self.selectedCat == None:
                idX = idx-1
            else:
                if self.classDict[idx-1] in self.selectedCat:
                    idX = self.selectedCatDict[self.classDict[idx-1]] 
                else:
                    continue
            # import ipdb;ipdb.set_trace()
            for _id,iterVal in enumerate(iterVals):
                classExtract[idX,:,:] += masks[idx-1,iterVal,:,:]


            classExtract[idX,:,:]=classExtract[idX,:,:]/len(iterVals) 
            classExtract[idX,:,:]=np.clip(classExtract[idX,:,:],0,1)
            # classExtract[idX,:,:] = (classExtract[idX,:,:]>0).astype(np.uint8)
            classExtract2d[idX,:,:,:] = np.dstack([classExtract[idX,:,:],classExtract[idX,:,:],classExtract[idX,:,:]])*RGBoGray.astype(np.uint8)

            combinedMask += classExtract[idX,:,:]
        
        combinedMask = np.clip(combinedMask,0,1)
        combinedRGB = np.dstack([combinedMask,combinedMask,combinedMask])* RGBoGray.astype(np.uint8)
        
       
        return combinedMask,combinedRGB, classExtract2d, classExtract

    def _getDistribution__(self):

        self.Dict={i:[] for i in self.selectedCatDict.values()}
        # import ipdb;ipdb.set_trace()
        for idx in self.ids:#range(self.__len__()):
            filterIds = self.COCOInstance.getAnnIds(idx)
            anns = self.COCOInstance.loadAnns(filterIds)
            classD,count = np.unique(np.array([i['category_id']-1 for i in anns]), return_counts = True)
            classID = classD[np.argmin(count)]
            self.Dict[classID].append(idx)

        # self.Indexdict={v:k for k,v in Classdict.items()}
        with open('Dict.json','w', encoding ='utf8') as f: 
            json.dump(self.Dict,f, indent = 6)
            f.close()
        DictCpy = {k:[] for k in self.Dict.keys()}
        for k,v in self.Dict.items():
            for j in v:
                DictCpy[k].append(j)
        return(DictCpy)
    
    # creates unique index(for 16-concecutive frame) from a unique video in a given class(key - activty class)
    def __get__(self, key, dict):
        while(True):
            if len(dict[key]) != 0:
                index = random.choice(dict[key])
                dict[key].remove(index)
                break
            else:
                dict[key] = [i for i in self.Dict[key]]
                continue

        return(index, dict)

    def __getitem__(self, i, selected_Class = True):
        # idx = self.ids[i]
        class_id = random.choice([i for i in self.selectedCatDict.values()])

        # if len(self.dict[class_id])==0:
        #     import ipdb;ipdb.set_trace()

        idx,self.dict = self.__get__(key = class_id, dict = self.dict)

        imgDetail = self.coco_imgs[idx]
        img_path = self.imgs_dir + '/' + imgDetail['file_name']


        imgCV = cv2.imread(img_path.replace('\\','/'))
        imgPIL = Image.fromarray(imgCV)
        imgPIL = imgPIL.resize(self.size[::-1])
        img = np.array(imgPIL)

        # classD = img_path.replace('\\','/').split('/')[::-1][1]

        filterIds = self.COCOInstance.getAnnIds(idx)
        anns = self.COCOInstance.loadAnns(filterIds)
        # classD = [i['category_id']-1 for i in anns]

        image_details = self.COCOInstance.loadImgs(anns[0]['image_id'])[0]
        h = image_details['height']
        w = image_details['width']
        classD = [i['category_id']-1 for i in anns]
        classLabels = [self.classDict[i] for i in classD]
        
        oneHot = np.zeros(len(self.selectedCat))
        for idx in classD:
            if self.classDict[idx] in self.selectedCat:
                idX = self.selectedCatDict[self.classDict[idx]] 
                oneHot[idX] = 1


        # import ipdb;ipdb.set_trace()

        masks, anns = annsToMask(anns, h, w, self.size[::-1])

        # masksPIL = Image.fromarray(masks)
        # masks = np.array(masksPIL.resize(self.size))

        activation = np.zeros((h, w)).astype('int16')

        anns_ids = [i['category_id'] for i in anns]
        maskFilterAll = self.getLevelMasks(masks, anns_ids, image = img, no_levels=4, GaussianFilter = True, RGB = True)
        # 
        anns_ids = [i for i in np.unique(np.array(anns_ids))];anns_ids.sort()

        combinedMask,combinedRGB, classExtract2d, classExtract = self.getCombinedMask(masks = maskFilterAll, anns_ids = anns_ids, iterVals=[0,1,2], image = img, RGB = True)
        # if len(anns_ids) >1:
        #     # import ipdb;ipdb.set_trace()
        #     os.makedirs('./image/',exist_ok  = True) 
        #     no=np.random.randint(50)
        #     cv2.imwrite('./image/a%d.jpg'%(no),cv2.vconcat([img, combinedRGB.astype(np.uint8), cv2.cvtColor((combinedMask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
        #     for idx,(y,x) in enumerate(zip(classExtract,classExtract2d)): 
        #         cv2.imwrite('./image/a%d-%d.jpg'%(no,idx),cv2.vconcat([img, x.astype(np.uint8), cv2.cvtColor((y*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
        # import ipdb;ipdb.set_trace()



        # img_file = glob(self.imgs_dir + idx + '.*')

        # assert len(img_file) == 1, \
        #     'Either no image or multiple images found for the ID {}: {}'.format(idx,img_file)
        # # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])

        # assert img.size == mask.size, \
        #     'Image and mask {} should be the same size, but are {} and {}'.format(idx,img.size,mask.size)

        # combinedMask = self.preprocess(combinedMask, self.scale)
        # combinedRGB = self.preprocess(combinedRGB, self.scale)

        if self.preProcess == None:
                
            # return(torch.from_numpy(img.copy()).type(torch.FloatTensor)/255.0, torch.from_numpy(combinedMask).type(torch.FloatTensor))

            return {
                'class' : classD,
                'inputImage' : torch.from_numpy(img.copy()).type(torch.FloatTensor)/255.0,
                'combinedMask': torch.from_numpy(combinedMask).type(torch.FloatTensor),
                'combinedRGB': torch.from_numpy(combinedRGB).type(torch.FloatTensor)/255.0,
                'classExtract': torch.from_numpy(classExtract).type(torch.FloatTensor),
                'classExtract2d': torch.from_numpy(classExtract2d).type(torch.FloatTensor)/255.0,
            }

        else:
            return {
                'class' : (classD, torch.tensor(oneHot, dtype = torch.long).unsqueeze(0)),
                'inputImage' : self.preProcess[0](img.copy()).unsqueeze(0),
                'combinedMask': self.preProcess[0]((combinedMask*255).astype(np.uint8)).unsqueeze(0),
                'combinedRGB': self.preProcess[0](combinedRGB.astype(np.uint8)).unsqueeze(0),
                'classExtract' : torch.cat([ self.preProcess[2]((i*255).astype(np.uint8)) for i in classExtract], dim=0).unsqueeze(0),
                'classExtract2d' : torch.cat([ self.preProcess[2]((i).astype(np.uint8)).unsqueeze(0) for i in classExtract2d], dim=0).unsqueeze(0)
                # 'classExtract': self.preProcess(torch.from_numpy(classExtract).type(torch.FloatTensor)),
                # 'classExtract2d': torch.from_numpy(classExtract2d),
            }



if __name__ == '__main__':

    folder = '/media/abraham/toFw/MS-COCO/annotations_trainval2017/annotations/'
    coco_instance = COCO(folder+'instanceSelectClassImage.json')
    imgs_dir = '/media/abraham/toFw/MS-COCO/train2017'

    selected_Class = ['person','truck','car','bus','train','motorcycle']
    size = (256,256)


    preprocess = transforms.Compose([ transforms.ToPILImage(), transforms.RandomVerticalFlip(0.3),transforms.RandomHorizontalFlip(0.3), transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0), transforms.ToTensor(),])
    preprocessOut = transforms.Compose([ transforms.ToPILImage(), transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0), transforms.ToTensor()])
    preprocess256 = transforms.Compose([ transforms.ToPILImage(), transforms.Resize((256,256)), transforms.ToTensor()])
    preprocess64 = transforms.Compose([ transforms.ToPILImage(), transforms.Resize((64,64)), transforms.ToTensor()])


    dataset = Simpload(COCOInstance = coco_instance, imgs_dir = imgs_dir, preProcess = [preprocess256,preprocess256,preprocess64], size=size, selectedCat = selected_Class)
    val_percent = 0.2
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    torch.manual_seed(0);
    train_set, test_set = random_split(dataset, [n_train, n_val])

    folder = './funyHuh'
    os.makedirs(folder,exist_ok=True)
    # import ipdb;ipdb.set_trace()

    print('Never been, never will ever ')
    dec = {i:0 for i in dataset.selectedCatDict.values()}

    for i_x,i in enumerate(tqdm(train_set, total = int(train_set.__len__()))): #np.random.randint(int(len(train_set)),size = 100)):
        ix = dataset[i] #np.random.randint(len(i['class']))
        for _id in np.array(np.unique(ix['class'][0])):
            dec[_id]+=1

        # imB = np.array(transforms.ToPILImage()(ix['combinedRGB'].squeeze()))
        # imI = np.array(transforms.ToPILImage()(ix['inputImage'].squeeze()))
        # imBW = np.array(transforms.ToPILImage()(ix['combinedMask'].squeeze()))
        # cvImage = cv2.hconcat([imI,imB,np.stack([imBW,imBW,imBW],axis=2)])
        # cv2.imwrite(folder+'/%d-imB.jpg'%i, cvImage)






