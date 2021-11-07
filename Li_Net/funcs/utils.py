#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:33:55 2019

@author: tao
"""
import os
import scipy.io as sio 
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from numpy.lib.stride_tricks import as_strided as ast
import torchvision.transforms.functional as F
from torchvision import models
import torch.nn as nn

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def preceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value_mean = 0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        feat = torch.abs(A_feat - B_feat)
        loss_value_mean = loss_value_mean + torch.mean(feat)
    return loss_value_mean


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    # total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((source - target) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算



def generate_2D_patches(in_data):
    
    # in_data     --> 240*240*155 for BraTS 
    # out_patch   --> 128*128*128
    # num_patches --> num = 9
    #in_size  = [240,240,155]
    in_size  = [in_data.shape[0], in_data.shape[1]]
    out_size = [128,128]
    num      = max(in_data.shape[0]//out_size[0] + 1, in_data.shape[1]//out_size[1] + 1)
    
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)

    count    = 0
    patches = list()

    for i in range(len(x_locs)):
        for j in range(len(y_locs)):
            xx = x_locs[i][0]
            yy = y_locs[j][0]

            tensor_img = F.to_tensor(in_data[xx:xx+out_size[0],yy:yy+out_size[1],:]) * 2 - 1.
            print(tensor_img)
            exit(0)
            patches.append(tensor_img)
            count = count + 1
                
    return patches  


def generate_all_2D_patches(in_data):
    
    #in_size  = [160,180]
    used_data   = in_data[:,:,:]
    out_size    = [128,128]
    num         = 1
    out_patches = list()

    for i in range(num):
        out_patches = out_patches + generate_2D_patches(used_data[:,:,:])
        
    return np.array(out_patches)


def generate_2D_pathches_slice_test(in_data):
    
    #in_size  = [160,180,155]
    used_data = in_data
    out_size  = [128,128]
    num       = used_data.shape[2]
    out_patches = np.zeros([num*4, out_size[0],out_size[1]])
    for i in range(num):
        out_patches[i*4:(i+1)*4] = generate_2D_patches(used_data[:,:,i])
        
    return out_patches

def generate_2D_patches_slice(in_data):
    
    #in_size  = [160,180]
    used_data = in_data
    out_size  = [128,128]
    num       = 1
    out_patches = np.zeros([num*4, out_size[0],out_size[1]])
    out_patches = generate_2D_patches(used_data[:,:])
        
    return out_patches


def generate_patch_loc(in_size,out_size,num):
    
    locs  = np.zeros([num,1])
    for i in range(num):
        if i == 0: 
            locs[i] = 0
        else:
            locs[i] = int((in_size-out_size)/(num-1))*i 
            
    #locs[i] = in_size-out_size - 1
    
    return locs.astype(int)

def prediction_in_testing_2Dimages(x_in,x03_real):
    
    
    x_in_re = torch.reshape(torch.reshape(x_in,[x_in.shape[0],x_in.shape[2],x_in.shape[3]]), [int(x_in.shape[0]/4),4,x_in.shape[2],x_in.shape[3]])
   
    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
     
    pred_images  = np.zeros([x_in_re.shape[0],in_size[0],in_size[1]])
    pred_values  = np.zeros([x_in_re.shape[0],3])
    
    for k in range(x_in_re.shape[0]):
        
        count  = 0
        matOut = torch.zeros((in_size[0],in_size[1]))
        used   = torch.zeros((in_size[0],in_size[1]))  
        
        cur_real_data = torch.reshape(x03_real[:,:,k],[160,180])
        cur_real_data = cur_real_data - cur_real_data.min()
        cur_real_data = cur_real_data/cur_real_data.max()
                
        for i in range(len(x_locs)):
            for j in range(len(y_locs)):
                xx = x_locs[i][0]
                yy = y_locs[j][0]
                        
                temp_out = x_in_re[k,count,:,:]
                temp_out = torch.reshape(temp_out,[128,128])
                        
                        
                # normalization
                temp_out = temp_out - temp_out.min()
                temp_out = temp_out/temp_out.max()
                        
                        
                matOut[xx:xx+out_size[0],yy:yy+out_size[1]] = matOut[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_out.cpu()
                used[xx:xx+out_size[0],yy:yy+out_size[1]]   = used[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
                
                count = count + 1
                
        #--------------------
        pred_res = matOut/used
        pred_res = pred_res - pred_res.min()
        pred_res = pred_res/pred_res.max()
        
        cur_real_data = cur_real_data - cur_real_data.min()
        cur_real_data = cur_real_data/cur_real_data.max()
        
        # --------------------
        psnr = compute_psnr(pred_res, cur_real_data)
        nmse = compute_nmse(pred_res, cur_real_data)
        ssim = compute_ssim(pred_res, cur_real_data)
                    
        pred_images[k,:,:] = (matOut/used).cpu().detach().numpy()
        pred_values[k,:]   = [psnr,nmse,ssim]
        
    return pred_images,pred_values



def prediction_in_testing_2DimagesNEW(pred_out,x03_real):
    #print(pred_out.shape)
    x_in = pred_out
    x_in_re = torch.reshape(torch.reshape(x_in,[x_in.shape[0],x_in.shape[2],x_in.shape[3]]), [int(x_in.shape[0]/4),4,x_in.shape[2],x_in.shape[3]])
   
    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
     
    pred_images  = np.zeros([x_in_re.shape[0],in_size[0],in_size[1]])
    pred_values  = np.zeros([x_in_re.shape[0],2])
    
    for k in range(x_in_re.shape[0]):
        
        count  = 0
        matOut = torch.zeros((in_size[0],in_size[1]))
        used   = torch.zeros((in_size[0],in_size[1]))  
        
        cur_real_data = torch.reshape(x03_real[:,:,k],[160,180])
        cur_real_data = cur_real_data - cur_real_data.min()
        
        cur_real_data = cur_real_data/cur_real_data.max()
#        print('our model:',[cur_real_data.max(),cur_real_data.min(),aa.max(),aa.min()])
                
        for i in range(len(x_locs)):
            for j in range(len(y_locs)):
                xx = x_locs[i][0]
                yy = y_locs[j][0]
                        
                temp_out = x_in_re[k,count,:,:]
                temp_out = torch.reshape(temp_out,[128,128])
                        
                        
                # normalization
                temp_out = temp_out - temp_out.min()
                temp_out = temp_out/temp_out.max()
                        
                        
                matOut[xx:xx+out_size[0],yy:yy+out_size[1]] = matOut[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_out.cpu()
                used[xx:xx+out_size[0],yy:yy+out_size[1]]   = used[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
                
                count = count + 1
                
        #--------------------
        pred_res = matOut/used +0.2
        pred_res = pred_res - pred_res.min()
        pred_res = pred_res/pred_res.max()
        
#        cur_real_data = cur_real_data - cur_real_data.min()
#        cur_real_data = cur_real_data/cur_real_data.max()
        
        # --------------------
        psnr = compute_psnr(pred_res, cur_real_data)
        nmse = compute_nmse(pred_res, cur_real_data)
       # ssims = compute_ssim(pred_res, cur_real_data)
                    
        pred_images[k,:,:] = (matOut/used).cpu().detach().numpy()
        pred_values[k,:]   = [psnr,nmse]
        #pred_values = 1
    return pred_images,pred_values



def prediction_syn_results(pred_out,real_out):
    
    ###########################################################################
    ### Note that
    # there is two manners to evaluate the testing sets
    # for example, using T1 + T2 to synthesize Flair
    # -->(1) the ground truths of Flair keep original size ([160,180,batch_size]) without spliting into small pathces (128*128). In this case, the 
    # synthesized results with size [batch_size*num_patch,1,128,128]， we need change it to [160,180,batch_size]
     
    # -->(2) the ground truths and synthesized results are all with size [batch_size*num_patch,1,128,128]， we need change 
    # them to [160,180,batch_size]. See details of this maner below.
    
    # When one volume as input, we set batch_size=num_slice
        
    ###########################################################################    

    
    # [batch_size*num_patch,1,128,128] -- > [batch_size, num_patch, 128, 128]
    pred_out_re = torch.reshape(torch.reshape(pred_out,[pred_out.shape[0],pred_out.shape[2],pred_out.shape[3]]), [int(pred_out.shape[0]/4),4,pred_out.shape[2],pred_out.shape[3]])
    real_out_re = torch.reshape(torch.reshape(real_out,[real_out.shape[0],real_out.shape[2],real_out.shape[3]]), [int(real_out.shape[0]/4),4,real_out.shape[2],real_out.shape[3]])

    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  # num_patch = num*num
    
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
     
    pred_images  = np.zeros([in_size[0],in_size[1],pred_out_re.shape[0]])
    real_images  = np.zeros([in_size[0],in_size[1],pred_out_re.shape[0]])
    
    for k in range(pred_out_re.shape[0]):
        
        count  = 0
        mat_pred_Out = torch.zeros((in_size[0],in_size[1]))
        used_pred    = torch.zeros((in_size[0],in_size[1]))  
        
        mat_real_Out = torch.zeros((in_size[0],in_size[1]))
        used_real    = torch.zeros((in_size[0],in_size[1]))  
     
        ## 
        for i in range(len(x_locs)):
            for j in range(len(y_locs)):
                xx = x_locs[i][0]
                yy = y_locs[j][0]
                        
                temp_pred_out = pred_out_re[k,count,:,:]
                temp_pred_out = torch.reshape(temp_pred_out,[128,128])
                        
                temp_real_out = real_out_re[k,count,:,:]
                temp_real_out = torch.reshape(temp_real_out,[128,128])
                        
                # normalization
                temp_pred_out = temp_pred_out - temp_pred_out.min()
                temp_pred_out = temp_pred_out/temp_pred_out.max()
                
                temp_real_out = temp_real_out - temp_real_out.min()
                temp_real_out = temp_real_out/temp_real_out.max()
                        
                        
                mat_pred_Out[xx:xx+out_size[0],yy:yy+out_size[1]] = mat_pred_Out[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_pred_out.cpu()
                used_pred[xx:xx+out_size[0],yy:yy+out_size[1]]    = used_pred[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
 
                mat_real_Out[xx:xx+out_size[0],yy:yy+out_size[1]] = mat_real_Out[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_real_out.cpu()
                used_real[xx:xx+out_size[0],yy:yy+out_size[1]]    = used_real[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
                
                
                count = count + 1
                
        #--------------------
        pred_res = mat_pred_Out/used_pred 
        real_res = mat_real_Out/used_real 
        
             
        pred_images[:,:,k] = pred_res.detach().numpy()#pred_res.cpu().detach().numpy()
        real_images[:,:,k] = real_res.detach().numpy() 
        
    
    pred_images = pred_images - pred_images.min()
    pred_images = pred_images/pred_images.max()
    
    real_images = real_images - real_images.min()
    real_images = real_images/real_images.max()  
    
    errors = ErrorMetrics(pred_images.astype(np.float32), real_images.astype(np.float32))  
        
    return errors


def loadSubjectData(path):
    
    data_imgs = sio.loadmat(path) 
    
    img_flair = data_imgs['data']['img_flair'][0][0].astype(np.float32)
    img_t1    = data_imgs['data']['img_t1'][0][0].astype(np.float32)
    img_t1ce  = data_imgs['data']['img_t1ce'][0][0].astype(np.float32)
    img_t2    = data_imgs['data']['img_t2'][0][0].astype(np.float32)
            
    return img_t1,img_t1ce,img_t2,img_flair
 


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    

class Logger(object):
	'''Save training process to log file with simple plot function.'''
	def __init__(self, fpath, title=None, resume=False): 
		self.file = None
		self.resume = resume
		self.title = '' if title == None else title
		if fpath is not None:
			if resume: 
				self.file = open(fpath, 'r') 
				name = self.file.readline()
				self.names = name.rstrip().split('\t')
				self.numbers = {}
				for _, name in enumerate(self.names):
					self.numbers[name] = []

				for numbers in self.file:
					numbers = numbers.rstrip().split('\t')
					for i in range(0, len(numbers)):
						self.numbers[self.names[i]].append(numbers[i])
				self.file.close()
				self.file = open(fpath, 'a')  
			else:
				self.file = open(fpath, 'w')

	def set_names(self, names):
		if self.resume: 
			pass
		# initialize numbers as empty list
		self.numbers = {}
		self.names = names
		for _, name in enumerate(self.names):
			self.file.write(name)
			self.file.write('\t')
			self.numbers[name] = []
		self.file.write('\n')
		self.file.flush()


	def append(self, numbers):
		assert len(self.names) == len(numbers), 'Numbers do not match names'
		for index, num in enumerate(numbers):
			self.file.write("{0:.6f}".format(num))
			self.file.write('\t')
			self.numbers[self.names[index]].append(num)
		self.file.write('\n')
		self.file.flush()

	def plot(self, names=None):   
		names = self.names if names == None else names
		numbers = self.numbers
		for _, name in enumerate(names):
			x = np.arange(len(numbers[name]))
			plt.plot(x, np.asarray(numbers[name]))
		plt.legend([self.title + '(' + name + ')' for name in names])
		plt.grid(True)

	def close(self):
		if self.file is not None:
			self.file.close()
             

class AverageMeter(object):
	"""Computes and stores the average and current value
	   Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		
	def avg(self):
		return self.sum / self.count

def mkdir_p(path):
	'''make dir if not exist'''
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise    
            

        
def model_task(inputs):
    x1 = torch.reshape(inputs[in_id1], [inputs[in_id1].shape[1]*inputs[in_id1].shape[0],1,inputs[in_id1].shape[2],inputs[in_id1].shape[3]]).type(torch.FloatTensor)
    x1 = x1.transpose(1, 3).transpose(2, 3).contiguous()

    x2 = torch.reshape(inputs[in_id2], [inputs[in_id2].shape[1]*inputs[in_id2].shape[0],1,inputs[in_id2].shape[2],inputs[in_id2].shape[3]]).type(torch.FloatTensor)
    x2 = x2.transpose(1, 3).transpose(2, 3).contiguous()

    return x1,x2

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def ErrorMetrics(vol_s, vol_t):

    # calculate various error metrics.
    # vol_s should be the synthesized volume (a 3d numpy array) or an array of these volumes
    # vol_t should be the ground truth volume (a 3d numpy array) or an array of these volumes

#    vol_s = np.squeeze(vol_s)
#    vol_t = np.squeeze(vol_t)

#    vol_s = vol_s.numpy()
#    vol_t = vol_t.numpy()

    assert len(vol_s.shape) == len(vol_t.shape) == 3
    assert vol_s.shape[0] == vol_t.shape[0]
    assert vol_s.shape[1] == vol_t.shape[1]
    assert vol_s.shape[2] == vol_t.shape[2]

    vol_s[vol_t == 0] = 0
    vol_s[vol_s < 0] = 0

    errors = {}

    vol_s = vol_s.astype(np.float32)

    # errors['MSE'] = np.mean((vol_s - vol_t) ** 2.)
    errors['MSE'] = np.sum((vol_s - vol_t) ** 2.) / np.sum(vol_t**2)
    errors['SSIM'] = ssim(vol_t, vol_s)
    dr = np.max([vol_s.max(), vol_t.max()]) - np.min([vol_s.min(), vol_t.min()])
    errors['PSNR'] = psnr(vol_t, vol_s, dynamic_range=dr)

#    # non background in both
#    non_bg = (vol_t != vol_t[0, 0, 0])
#    errors['SSIM_NBG'] = ssim(vol_t[non_bg], vol_s[non_bg])
#    dr = np.max([vol_t[non_bg].max(), vol_s[non_bg].max()]) - np.min([vol_t[non_bg].min(), vol_s[non_bg].min()])
#    errors['PSNR_NBG'] = psnr(vol_t[non_bg], vol_s[non_bg], dynamic_range=dr)
#
#    vol_s_non_bg = vol_s[non_bg].flatten()
#    vol_t_non_bg = vol_t[non_bg].flatten()
#
#    # errors['MSE_NBG'] = np.mean((vol_s_non_bg - vol_t_non_bg) ** 2.)
#    errors['MSE_NBG'] = np.sum((vol_s_non_bg - vol_t_non_bg) ** 2.) /np.sum(vol_t_non_bg**2)

    return errors
