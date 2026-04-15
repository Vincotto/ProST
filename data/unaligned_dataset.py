import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from data.constraint_dataset import ConstraintDataset


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    # def __getitem__(self, index):
    #     """Return a data point and its metadata information.

    #     Parameters:
    #         index (int)      -- a random integer for data indexing

    #     Returns a dictionary that contains A, B, A_paths and B_paths
    #         A (tensor)       -- an image in the input domain
    #         B (tensor)       -- its corresponding image in the target domain
    #         A_paths (str)    -- image paths
    #         B_paths (str)    -- image paths
    #     """
    #     A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
    #     if self.opt.serial_batches:   # make sure index is within then range
    #         index_B = index % self.B_size
    #     else:   # randomize the index for domain B to avoid fixed pairs.
    #         index_B = random.randint(0, self.B_size - 1)
    #     B_path = self.B_paths[index_B]
    #     A_img = Image.open(A_path).convert('RGB')
    #     B_img = Image.open(B_path).convert('RGB')
    #     # apply image transformation
    #     A = self.transform_A(A_img)
    #     B = self.transform_B(B_img)

    #     return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    # def __getitem__(self, index):
    #     """Return a data point and its metadata information."""
    #     A_path = self.A_paths[index % self.A_size]
        
    #     if self.opt.serial_batches:   # make sure index is within then range
    #         index_B = index % self.B_size
    #     else:   # randomize the index for domain B to avoid fixed pairs.
    #         index_B = random.randint(0, self.B_size - 1)
    #     B_path = self.B_paths[index_B]
        
    #     A_img = Image.open(A_path).convert('RGB')
    #     B_img = Image.open(B_path).convert('RGB')
        
    #     # Apply image transformation
    #     A = self.transform_A(A_img)
    #     B = self.transform_B(B_img)
        
    #     result = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        
    #     # 如果启用约束数据集
    #     if hasattr(self, 'constraint_dataset') and self.constraint_dataset is not None:
    #         # 方式1：根据A的文件名寻找对应的C图片
    #         A_filename = os.path.basename(A_path)
    #         A_name_without_ext = os.path.splitext(A_filename)[0]
            
    #         # 在C数据集中寻找相同文件名的图片
    #         C_path = None
    #         for c_path in self.constraint_dataset.C_paths:
    #             c_filename = os.path.basename(c_path)
    #             c_name_without_ext = os.path.splitext(c_filename)[0]
    #             if c_name_without_ext == A_name_without_ext:
    #                 C_path = c_path
    #                 break
            
    #         if C_path is not None:
    #             C_img = Image.open(C_path).convert('RGB')
    #             C = self.constraint_dataset.transform_C(C_img)
    #             result['C'] = C
    #             result['C_paths'] = C_path
    #         else:
    #             # 如果找不到对应文件名，回退到随机选择
    #             print(f"Warning: No matching constraint image found for {A_filename}, using random selection")
    #             constraint_data = self.constraint_dataset[random.randint(0, len(self.constraint_dataset) - 1)]
    #             result.update(constraint_data)
        
    #     return result

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        A_path = self.A_paths[index % self.A_size]
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        
        result = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        
        # 处理约束图像C - 优化版本
        if hasattr(self, 'constraint_dataset') and self.constraint_dataset is not None:
            # 直接根据A的文件名构造C的路径
            A_filename = os.path.basename(A_path)
            A_name_without_ext = os.path.splitext(A_filename)[0]
            
            # 直接构造C图像的路径
            C_path = os.path.join(self.constraint_dataset.dir_C, A_filename)
            
            # 如果直接路径不存在，尝试其他常见扩展名
            if not os.path.exists(C_path):
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    C_path_try = os.path.join(self.constraint_dataset.dir_C, A_name_without_ext + ext)
                    if os.path.exists(C_path_try):
                        C_path = C_path_try
                        break
            
            if os.path.exists(C_path):
                C_img = Image.open(C_path).convert('RGB')
                C = self.constraint_dataset.transform_C(C_img)
                result['C'] = C
                result['C_paths'] = C_path
            else:
                print(f"Warning: No matching constraint image found for {A_filename}")
                # 可选：回退到随机选择或者不添加C
                # constraint_data = self.constraint_dataset[random.randint(0, len(self.constraint_dataset) - 1)]
                # result.update(constraint_data)
        
        return result


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
