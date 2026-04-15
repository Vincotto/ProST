# data/constraint_dataset.py
import os
from PIL import Image
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

class ConstraintDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        # A目录（草图）和C目录（色彩图像）
        self.dir_A = opt.dataroot + '/trainA'  # 或者从opt中获取
        self.dir_C = opt.dataroot + '/trainC'   # 色彩图像目录
        
        # 获取所有文件路径
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        
        # 创建文件名到路径的映射
        self.C_path_dict = {}
        for c_path in self.C_paths:
            filename = os.path.basename(c_path)
            # 去掉扩展名作为key，或者保留完整文件名
            name_without_ext = os.path.splitext(filename)[0]
            self.C_path_dict[name_without_ext] = c_path
        
        self.transform_C = get_transform(opt, grayscale=False)
    
    def get_constraint_for_A(self, A_path):
        A_filename = os.path.basename(A_path)
        A_name = os.path.splitext(A_filename)[0]
        
        # 查找对应的C图像
        if A_name in self.C_path_dict:
            C_path = self.C_path_dict[A_name]
            C_img = Image.open(C_path).convert('RGB')
            C = self.transform_C(C_img)
            return C, C_path
        else:
            # 如果找不到对应的C，可以返回None或者抛出异常
            print(f"Warning: No constraint image found for {A_filename}")
            return None, None
    
    def __len__(self):
        return len(self.A_paths)
    
    def __getitem__(self, index):
        A_path = self.A_paths[index]
        C, C_path = self.get_constraint_for_A(A_path)
        return {'C': C, 'C_paths': C_path}
