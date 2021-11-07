from LiNet_SynthModel import LatentSynthModel
from config import opt
import fire

    
def train(**kwargs):
    KD = opt.KD
    model_id = opt.model_id
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)

    if KD:
        SynModel.train()
        SynModel.train_one_model(True, model_id)
    else:
        SynModel.train_one_model(False, model_id)
   
if __name__ == '__main__':
    
    fire.Fire()