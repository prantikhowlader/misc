import numpy as np
import os
from collections import defaultdict
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
mao_path = '/scratch/KurcGroup/mazhao/wsi_prediction_png/pred_out_6_slides_300/19test_10x20xcomb_v2/'
shahira_path = '/scratch/KurcGroup/phowlader/test_union/shahira/new_test_set19_out_threshcomp_formatmz/'
mao_file = 'L28352-multires_Image_978_ratio1.0_wsi_argmax.png'
shahira_file = 'L28352-multires_Image_978_ratio1.0_wsi.npy'
mname = mao_file.split('.')[0]
sname = shahira_file.split('.')[0]
mao_path = mao_path + mao_file
shahira_path = shahira_path+shahira_file
mao_numpy = cv2.imread(mao_path,0)
print(mao_numpy.shape)
sha_numpy = np.load(shahira_path)
print(sha_numpy.shape)


'''
sh
0: None
1:'K17_Pos'
2:'CD8-Purple'  
3:'CD16-Black'  
4:'CD4-Cyan'  
5:'CD3-Yellow'
6:'CD20-Red'
7:'Hematoxilin'  
8:'K17_Neg'
    
[1:CD16, 2:CD20, 3:CD3, 4:CD4, 5:CD8, 6:K17+, 7:K17-, 8:background]
''' 
mask_list = []
for i in np.unique(mao_numpy):
	#shahira_change
	if(i ==1):
		a = i
	if(i == 2):
		a = i
	if(i ==3):
		a =i 
	if(i ==4):
		a = i
	if(i == 5):
		a = i
	if(i ==6):
		a = i
	if(i ==7):
		a = i
	if(i ==8):
		a = i
	mask = (mao_numpy==i).astype('uint8')+(sha_numpy==a).astype('uint8')
	mask[mask==2]	= 1
	mask[mask==1] = i
	print(np.unique(mask))
	mask_list.append(mask)
print(len(mask_list))
check = 1
for j,mask in enumerate(mask_list):
	if(check == 1):
		fin_mask = mask
		check =0
	else:
		mask = mask * (1/(j+1))
		mask = np.rint(mask)
		fin_mask = fin_mask + mask
		
intersection = (fin_mask>1).astype('uint8')
check =1
for j,mask in enumerate(mask_list):
	if(check == 1):
		fin_mask = mask
		check =0
	else:
		#mask = mask * (1/(j+1))
		#mask = np.rint(mask)
		fin_mask = fin_mask + mask
fin_mask = fin_mask - np.multiply(fin_mask,intersection)
fin_mask_shahira = fin_mask + np.multiply(sha_numpy,intersection)
fin_mask_shahira[fin_mask_shahira==0]= 8
fin_mask_mao = fin_mask + np.multiply(mao_numpy,intersection)
print(np.unique(fin_mask_shahira))
print(np.unique(fin_mask_mao))
np.save(mname+'_mao',fin_mask_mao)
np.save(sname+'_sha',fin_mask_shahira)

for i in np.unique(fin_mask_mao):
	plt.imshow((fin_mask_mao==i).astype('uint8'))
	plt.axis("off")
	plt.savefig('./' + mname+'_'+str(i)+'_M.png', bbox_inches='tight',dpi = 1000)
	plt.close()
for i in np.unique(fin_mask_shahira):
	plt.imshow((fin_mask_shahira==i).astype('uint8'))
	plt.axis("off")
	plt.savefig('./' + sname+'_'+str(i)+'_S.png', bbox_inches='tight',dpi = 1000)
	plt.close()
for i in np.unique(mao_numpy):
	plt.imshow((mao_numpy==i).astype('uint8'))
	plt.axis("off")
	plt.savefig('./' + mname+'_'+str(i)+'_MOSLIDE.png', bbox_inches='tight',dpi = 1000)
	plt.close()
for i in np.unique(sha_numpy):
	plt.imshow((sha_numpy==i).astype('uint8'))
	plt.axis("off")
	plt.savefig('./' + sname+'_'+str(i)+'_SOSLIDE.png', bbox_inches='tight',dpi = 1000)
	plt.close()


		 
