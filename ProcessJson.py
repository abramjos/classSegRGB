import json
from pycocotools.coco import COCO

with open('instances_train2017.json', 'r') as f:
	coco_instance = json.load(f)


labels = {1:'person',2:'truck',3:'car',4:'bus',5:'train',6:'motorcycle'}
labelsInv = {v:k for k,v in labels.items()}

# availCat = coco_instance['categories'][:]

# for i in range(coco_instance['categories']):
# 	if i['name'] in labels.values():
# 		i['id'] = int(labelsInv[i['name']])
# 	else:
# 		coco_instance['categories'].remove(i)

# coco_instance['annotations'] 



idx = 0
mapId = {}
while(True):
	if idx<len(coco_instance['categories']):
		i = coco_instance['categories'][idx]
		idx+=1
		
		if i['name'] in labels.values():
			mapId[i['id']] = int(labelsInv[i['name']])
			i['id'] = int(labelsInv[i['name']])
		else:
			coco_instance['categories'].remove(i)
			idx-=1
	else:
		break


idx = 0
imageAnno={}
while(True):
	if idx<len(coco_instance['annotations']):
		i = coco_instance['annotations'][idx]
		idx+=1
		
		if i['category_id'] in mapId.keys():
			i['category_id'] = mapId[i['category_id']]
			if i['image_id'] not in imageAnno.keys() 
				imageAnno[i['image_id']] = [i['category_id']]	
			else:
				imageAnno[i['image_id']].append(i['category_id'])
		else:
			coco_instance['annotations'].remove(i)
			idx-=1
	else:
		break


### IF i  cant find the imageAnno values 

idx=0
imageAnno={}
while(True):
	if idx<len(coco_instance['annotations']):
		i = coco_instance['annotations'][idx]
		idx+=1
		if i['category_id'] in mapId.values():
			if i['image_id'] not in imageAnno.keys(): 
				imageAnno[i['image_id']] = [i['category_id']]	
			else:
				imageAnno[i['image_id']].append(i['category_id'])
		else:
			coco_instance['annotations'].remove(i)
			idx-=1
	else:
		break

# ### #### Once I find imageAnno>?
idx=0
while(True):
	if idx<len(coco_instance['images']):
		imID = coco_instance['images'][idx]['id']

		if imID not in imageAnno.keys():
			coco_instance['images'].remove(coco_instance['images'][idx])
			idx-=1
		idx+=1
	else:
		break













######################################################################

GETTING STATS

######################################################################

# Getting imageAnno[image ID : classes]

imageAnno = {}
idx=0
while(True):
    if idx<len(coco_instance['images']):
        imID = coco_instance['images'][idx]['id']
        
        if imID not in imageAnno.keys():
            coco_instance['images'].remove(coco_instance['images'][idx])
            idx-=1
        idx+=1
    else:
        break


# calculate unique ones
#  class/img
count = {i:0 for i in labels.values()}
for k,v in imageAnno.items():
    for l in labels.keys():
        if l in v:
            count[labels[l]]+=1

