import os
import numpy as np 
import matplotlib.pyplot as plt 



def parse(log_data, logs = ['epoch','accuracy','lossRGB','loss_classif']):
	train = {i:[] for i in logs}
	test = {i:[] for i in logs}

	for line in log_data:
		line = line.replace('\n','').replace('\t','').replace(':','-').replace(' ','')
		for ix,section in enumerate(line.split('|')):
			try:
				if section == '':continue
				if section.split('-')[1] == 'Train':
					train['epoch'].append(int(section.split('-')[0]))
					trainF = True
					continue
				if section.split('-')[1] == 'Test':
					test['epoch'].append(int(section.split('-')[0]))
					trainF = False
					continue

				if section.split('-')[0]in logs:
					if trainF:
						train[section.split('-')[0]].append(float(section.split('-')[1]))
					else:
						test[section.split('-')[0]].append(float(section.split('-')[1]))
			except:
				import ipdb;ipdb.set_trace()
	return(train,test)

def convert_vals(data, logs = ['RAM','SWAP','CPU','EMC_FREQ','GR3D_FREQ','thermal_GPU','thermal_CPU',
	                 'thermal_thermal','VDD_CPU_GPU_CV','VDD_SOC']):
	numeric={i:[] for i in logs}
	for _id,val in zip(data.keys(),data.values()):
		for value in val:
			# print(value)
			if _id == 'CPU':
				numeric[_id].append([i for i in value.replace('[','').replace(']','').replace("'",'').split(',') if i!='off'])
			elif _id in ['thermal_GPU','thermal_CPU','thermal_thermal']:
				numeric[_id].append(float(value.replace('C','')))
			elif _id in ['RAM','SWAP','VDD_CPU_GPU_CV','VDD_SOC']:
				numeric[_id].append(value.split('/')[0])
	return(numeric)


# log_files = ['logs/tegra_log_normal.txt', 'logs/tegra_log_normal.txt', 'logs/tegra_log_model.txt','logs/tegra_log_trt.txt']
# log_data = [open(i,'r').readlines() for i in log_files]


log_files = ['./log/best.txt','./log/best18.txt','./log/best50.txt']

log_data = [open(i,'r').readlines() for i in log_files]
trainA, testA = parse(log_data[0][3:])
trainres18, testres18 = parse(log_data[1][3:])
trainres50, testres50 = parse(log_data[2][3:])

##############################################################################################
                                  # PLOTING
##############################################################################################
# Thermal

# len_max= len(trainA['epoch'])

# color = [['r','r--'],['g','g--'],['b','b']]
# for i,c in zip(['accuracy','loss_classif'],color):

# 	for idx,((train,test),(m1,m2)) in enumerate(zip([[trainA, testA], [trainres18, testres18], [trainres50, testres50]],[[".","o"],["1","2"],["h","H"]])):
# 		if idx in [1,2]:
# 			dataTrain = np.array(train[i])[0:len_max:2]
# 			dataTest = np.array(test[i])[0:len_max:2]
# 		# else:
# 		# 	dataTrain = np.array(train[i])[0:int(len_max/2.0)]
# 		# 	dataTest = np.array(test[i])[0:int(len_max/2.0)]


# 		plt.plot(dataTrain, c[0], marker=m1, label='Train-%s'%i)
# 		plt.plot(dataTest, c[0], marker=m2, label='Test-%s'%i, linestyle='dashed')


# plt.xlabel('Epoch') 
# plt.ylabel('Accuracy/Loss') 
# plt.title('Accuracy/Loss') 
# plt.legend()
# plt.show()



len_max= len(trainA['epoch'])
figure, axis = plt.subplots(2, 2)

metric_label = ['Accuracy','Loss']
archi_label = ['AE','Res18','Res50']
tLabel = ['Train','Test']
color = [['r','g','b'],['r--','g--','b--']]
dash=[[6, 2],[2, 2, 10, 2]]

f=10
for ixx,i in enumerate(['accuracy','loss_classif']):
	for idx,((train,test),(m)) in enumerate(zip([[trainA, testA], [trainres18, testres18], [trainres50, testres50]],[[".","o"],["1","2"],["h","H"]])):
		if idx in [0]:
			dataTrain = np.array(train[i])[0:len_max:2]
			dataTest = np.array(test[i])[0:len_max:2]
		else:
			dataTrain = np.array(train[i])[0:int(len_max/2.0)]
			dataTest = np.array(test[i])[0:int(len_max/2.0)]
		archi = archi_label[idx]
		label = metric_label[ixx]

		data = [dataTrain,dataTest]
		if i == 'accuracy':
			data=(np.array(data)*100).astype(np.uint8)
		# print('')
		for iii in [0,1]:
			c=color[iii][idx]
			TrTs = tLabel[iii]
			labelF = "%s-%s-%s"%(archi,TrTs,label)
			# axis[ixx, iii].set_title(labelF)
			axis[ixx, iii].plot(data[iii][::f], c, marker=m[iii], label=labelF, dashes=dash[iii])#, linestyle='dashed')
			# axis[ixx, iii].plot(dataTrain, c[0], marker=m1, label='Train-%s'%label)
			axis[ixx, iii].set_title("%s-%s"%(TrTs,label)) 
			axis[ixx, iii].set_xlabel('Epoch') 
			axis[ixx, iii].set_ylabel(metric_label[ixx]) 
			axis[ixx, iii].legend()

plt.show()
