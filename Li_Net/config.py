#coding:utf8
import warnings
class DefaultConfig(object):
    train_rate      = 0.7    ### 训练数据占总数据的百分比
    model_id        = 1      ### 选择训练model1对应的单模态分类网络
    KD              = True   ###True:以知识蒸馏的方式训练单模态分类网络  Flase:以原始方式训练单模态分类网络
    data_path       = 'fadata/'
    save_path       = 'results/'

    use_gpu         = True # user GPU or not    

    multimodel_lr   = 0.0005
    single_lr       = 0.0001 # initial learning rate
    lr_decay        = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay    = 1e-4 # 
    
    batch_size      = 3 # batch size
    epochs          = 200
    decay_epoch     = 100
    dropout         = 0.5
    
    fold            = 0
    lambda1         = 5
    lambda2         = 10
 
    checkpoint          = 'checkpointNEW'
    checkpoint_interval = 5  
    
    b1 = 0.5
    b2 = 0.999
    n_critic = 5

def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))

DefaultConfig.parse = parse
opt =DefaultConfig()

