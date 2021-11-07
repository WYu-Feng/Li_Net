
# Li-Net

Code for "Li-Net: LiDan Network for Multi-modal MR Image Classification"


----------------
Usage

1. The original implementation of Li-Net is Pytorch.
2. To run the code, you should first install dependencies:

   pip install fire
        
3. Setup all parameters in config.py

4. Put your data into ./dataset (Some samples from fadata have been stored out in this file, 
   'class2_1' means the 1 mode of the 2 type of data.)

5. Train
   
   python main.py train
   
   (you can set your parameters when runing the code)
   
[1] List of our team members: Dan Li, Lin Ding, Yufeng Wang, Cong Xu, Jiahao Li, Yihe Liu