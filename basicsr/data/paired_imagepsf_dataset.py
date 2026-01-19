from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.physics_transforms import abs_random_crop, pal_random_crop, palsr_random_crop, fov_augment, abs_random_crop_cor
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torch.nn.functional as F
import torch
import cv2
import random
from scipy.linalg import orth
import os
import multiprocessing
from PIL import Image
import torch.nn as nn
import csv


def restore_psfmap_from_table(table, H=1280, W=1920):
    n_field, n_angle, feat_dim = table.shape
    center = np.array([H // 2, W // 2])
    max_dist = np.linalg.norm(center)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    rel = np.stack([yy - center[0], xx - center[1]], axis=-1)
    dist = np.linalg.norm(rel, axis=-1)
    field_idx_map = ((dist / max_dist) * (n_field - 1)).astype(int)
    field_idx_map[dist == 0] = 0
    angle_map = np.degrees(np.arctan2(rel[..., 1], rel[..., 0]))
    angle_map = (angle_map + 360) % 360
    angle_idx_map = ((angle_map / 360) * n_angle).astype(int) % n_angle
    big_map = table[field_idx_map, angle_idx_map]  # [H, W, 78]
    return big_map

def lookup_psfmap_patch(table, top, left, patch_h=256, patch_w=256, H=1280, W=1920):
    n_field, n_angle, feat_dim = table.shape
    center = np.array([H // 2, W // 2])
    max_dist = np.linalg.norm(center)
    yy, xx = np.meshgrid(np.arange(top, top+patch_h), np.arange(left, left+patch_w), indexing='ij')
    rel = np.stack([yy - center[0], xx - center[1]], axis=-1)
    dist = np.linalg.norm(rel, axis=-1)
    field_idx_map = ((dist / max_dist) * (n_field - 1)).astype(int)
    field_idx_map[dist == 0] = 0
    angle_map = np.degrees(np.arctan2(rel[..., 1], rel[..., 0]))
    angle_map = (angle_map + 360) % 360
    angle_idx_map = ((angle_map / 360) * n_angle).astype(int) % n_angle
    patch_map = table[field_idx_map, angle_idx_map]  # [patch_h, patch_w, 78]
    return patch_map

def paired_paths_from_csv(csv_path):
    """
    从csv读取配对路径，返回与paired_paths_from_folder兼容的list[dict]，每个dict包含'lq_path'和'gt_path'。
    """
    paths = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append({'lq_path': row['lq_path'], 'gt_path': row['gt_path']})
    return paths 

def get_dark_channel(I, w):
    
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc

def get_category(value):
    if value < 0.2:  # 视场角 1
        return 0
    elif value < 0.4:  # 视场角 2
        return 1
    elif value < 0.6:  # 视场角 3
        return 2
    elif value < 0.8:  # 视场角 4
        return 3
    else:  # 视场角 5
        return 4
    # return int(value // 0.2)  # 将值划分为5类，0-1划分为[0, 0.2), [0.2, 0.4), ..., [0.8, 1)

def generate_weights(matrix):
# 找到最小值和最大值
    min_value = matrix.min()
    max_value = matrix.max()
    # print(min_value)
    # print(max_value)
    # 根据值确定类别


    # 获取最小值和最大值对应的类别
    min_category = get_category(min_value)
    max_category = get_category(max_value)

    # 确定最终的权重向量
    weights = torch.zeros(5)
    for category in range(min_category, max_category + 1):
        weights[category] = 1

    return weights
    # # print("最小值类别:", min_category.item())
    # # print("最大值类别:", max_category.item())
    # # print("权重向量:", weights)

    # # 假设这是你的5x3x11的tensor
    # tensor = torch.randn(5, 3, 11)

    # # 使用权重向量去mask tensor
    # # 将权重向量扩展到 (5, 1, 1) 以便进行广播
    # weights_expanded = weights.view(5, 1, 1)

    # # 进行元素级别的乘法
    # masked_tensor = tensor * weights_expanded

    # print("原始张量:", tensor)
    # print("Masked 张量:", masked_tensor)

def load_tensors_from_folder(folder_path):
    tensor_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            file_path = os.path.join(folder_path, filename)
            tensor = torch.load(file_path).cpu()
            
            tensor_dict[filename] = tensor

            # print(f"Warning: {filename} does not have the shape (5, 3, 11)")
    
    return tensor_dict


def transform_abd(folder_path, fov):
    tensor_dict = {}
    fov_binned = torch.clamp((fov * 5).to(torch.int64), 0, 4)
    height, width = fov.shape
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            file_path = os.path.join(folder_path, filename)
            tensor = torch.load(file_path).cpu()
            f, w, z = tensor.shape
            abd_map = torch.zeros((height, width, w*z))
            tensor = tensor.view(f, w*z)
            y_indices, x_indices = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            fov_indices = fov_binned[y_indices, x_indices]

            # 使用torch的高级索引直接填充tensor
            abd_map[y_indices, x_indices] = tensor[fov_indices]            

            # print(abd_map)
            tensor_dict[filename] = abd_map

            # print(f"Warning: {filename} does not have the shape (5, 3, 11)")
    
    return tensor_dict

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

@DATASET_REGISTRY.register()
class PairedImageDataset_psfabd_readcsv(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset_psfabd_readcsv, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.csv_path = opt['csv_path']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.label_root = opt['dataroot_label']
        self.abd_root = opt['dataroot_abd']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.paths = paired_paths_from_csv(self.csv_path)
        # if self.io_backend_opt['type'] == 'lmdb':
        #     self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        #     self.io_backend_opt['client_keys'] = ['lq', 'gt']
        #     self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #                                                   self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            # print(self.paths[0]['gt_path'])
            # print(self.paths[1]['gt_path'])
        self.abd = load_tensors_from_folder(self.abd_root)

        #generate position map
        #获取一张图像尺寸信息，生成位置图
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        lq_path = self.paths[0]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        H, W, C = img_lq.shape
        center_y, center_x = H // 2, W // 2

# 创建网格坐标
        y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # 计算每个点到中心的距离
        distances = torch.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

        # 最大值归一化
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        # 将 tensor 的值从 [0, 1] 范围转换为 [0, 255]
# 随机选择裁剪区域的左上角坐标
        # x = random.randint(0, 1920 - 256)
        # y = random.randint(0, 1280 - 256)

        # # 裁剪 256x256 的区域
        # crop_tensor = normalized_distances[y:y + 256, x:x + 256]

        # # 将 tensor 的值从 [0, 1] 范围转换为 [0, 255]
        # crop_tensor = (crop_tensor * 255).byte()

        # # 转换为 PIL 图像
        # image = Image.fromarray(crop_tensor.numpy(), mode='L')

        # # 保存图像
        # image.save('cropped_image.png')  # 替换为你想要的文件名
        self.fov_map = normalized_distances.unsqueeze(0)
        # normalized_distances 现在是一个新的点阵，包含归一化后的距离
        # print(normalized_distances.shape)  # 输出 torch.Size([1920, 1280])
        # x_p = (torch.range(0, H-1)/(H-1)-0.5)/0.5
        # y_p = (torch.range(0, W-1)/(W-1)-0.5)/0.5
        # x_p = (torch.range(0, H-1)/(H-1))
        # y_p = (torch.range(0, W-1)/(W-1))
        # fov = torch.zeros([2,H,W])
        # for i in range(H):
        #     fov[0, i, :] = y_p
        # for j in range(W):
        #     fov[1, :, j] = x_p

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # multiprocessing.set_start_method('spawn', force=True)
        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_name = os.path.basename(lq_path)
        txt_name = os.path.splitext(img_name)[0]+'.txt'
        label_path = os.path.join(self.label_root, txt_name)
        if self.opt['phase'] == 'train':
            with open(label_path, 'r') as file:
                # 读取文件中的路径，并添加到列表中
                label_name = file.readline().strip()  # 读取第一行并去掉换行符
            abd_tensor = self.abd[label_name]
            # print(abd_tensor.type())
        else:
            with open(label_path, 'r') as file:
                # 读取文件中的路径，并添加到列表中
                label_name = file.readline().strip()  # 读取第一行并去掉换行符
            abd_tensor = self.abd[label_name]

        
        # abd_path = os.path.join(self.abd_root, label_name)
        # print(abd_tensor)
        # abd = torch.load(abd_path)
        # print(abd_tensor.shape)
        fov = self.fov_map
        weights_expanded = 0
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, fov, top, left = abs_random_crop_cor(img_gt, img_lq, fov, gt_size, scale, gt_path)
            # flip, rotation
            [img_gt, img_lq], fov = fov_augment([img_gt, img_lq], fov, self.opt['use_flip'], self.opt['use_rot'])

            fov_mask = generate_weights(fov)
            # print(fov_mask)
            weights_expanded = fov_mask.view(5, 1, 1)
            # abd_tensor = abd_tensor * weights_expanded
            psfmap = lookup_psfmap_patch(abd_tensor, top, left).permute(2,0,1)
            # perturbation = abd_tensor * 0.25 * (torch.rand(abd_tensor.size()) * 2 - 1)
            # abd_tensor = abd_tensor + perturbation
        else:
            psfmap = restore_psfmap_from_table(abd_tensor).permute(2,0,1)
            # print(psfmap.shape)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'abd': psfmap,  'fov_mask':  weights_expanded, 'fov': fov, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



@DATASET_REGISTRY.register()
class PairedImageDataset_psfabd(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset_psfabd, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.label_root = opt['dataroot_label']
        self.abd_root = opt['dataroot_abd']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            # print(self.paths[0]['gt_path'])
            # print(self.paths[1]['gt_path'])
        self.abd = load_tensors_from_folder(self.abd_root)

        #generate position map
        #获取一张图像尺寸信息，生成位置图
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        lq_path = self.paths[0]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        H, W, C = img_lq.shape
        center_y, center_x = H // 2, W // 2

# 创建网格坐标
        y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # 计算每个点到中心的距离
        distances = torch.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

        # 最大值归一化
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        # 将 tensor 的值从 [0, 1] 范围转换为 [0, 255]
# 随机选择裁剪区域的左上角坐标
        # x = random.randint(0, 1920 - 256)
        # y = random.randint(0, 1280 - 256)

        # # 裁剪 256x256 的区域
        # crop_tensor = normalized_distances[y:y + 256, x:x + 256]

        # # 将 tensor 的值从 [0, 1] 范围转换为 [0, 255]
        # crop_tensor = (crop_tensor * 255).byte()

        # # 转换为 PIL 图像
        # image = Image.fromarray(crop_tensor.numpy(), mode='L')

        # # 保存图像
        # image.save('cropped_image.png')  # 替换为你想要的文件名
        self.fov_map = normalized_distances.unsqueeze(0)
        # normalized_distances 现在是一个新的点阵，包含归一化后的距离
        # print(normalized_distances.shape)  # 输出 torch.Size([1920, 1280])
        # x_p = (torch.range(0, H-1)/(H-1)-0.5)/0.5
        # y_p = (torch.range(0, W-1)/(W-1)-0.5)/0.5
        # x_p = (torch.range(0, H-1)/(H-1))
        # y_p = (torch.range(0, W-1)/(W-1))
        # fov = torch.zeros([2,H,W])
        # for i in range(H):
        #     fov[0, i, :] = y_p
        # for j in range(W):
        #     fov[1, :, j] = x_p

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # multiprocessing.set_start_method('spawn', force=True)
        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_name = os.path.basename(lq_path)
        txt_name = os.path.splitext(img_name)[0]+'.txt'
        label_path = os.path.join(self.label_root, txt_name)
        if self.opt['phase'] == 'train':
            with open(label_path, 'r') as file:
                # 读取文件中的路径，并添加到列表中
                label_name = file.readline().strip()  # 读取第一行并去掉换行符
            abd_tensor = self.abd[label_name]
            # print(abd_tensor.type())
        else:
            with open(label_path, 'r') as file:
                # 读取文件中的路径，并添加到列表中
                label_name = file.readline().strip()  # 读取第一行并去掉换行符
            abd_tensor = self.abd[label_name]

        
        # abd_path = os.path.join(self.abd_root, label_name)
        # print(abd_tensor)
        # abd = torch.load(abd_path)
        # print(abd_tensor.shape)
        fov = self.fov_map
        weights_expanded = 0
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            hh = gt_size
            ww = gt_size
            # random crop
            img_gt, img_lq, fov, top, left = abs_random_crop_cor(img_gt, img_lq, fov, gt_size, scale, gt_path)
            # flip, rotation
            [img_gt, img_lq], fov = fov_augment([img_gt, img_lq], fov, self.opt['use_flip'], self.opt['use_rot'])

            fov_mask = generate_weights(fov)
            # print(fov_mask)
            weights_expanded = fov_mask.view(5, 1, 1)
            # abd_tensor = abd_tensor * weights_expanded
            psfmap = lookup_psfmap_patch(abd_tensor, top, left, hh, ww).permute(2,0,1)
            # perturbation = abd_tensor * 0.25 * (torch.rand(abd_tensor.size()) * 2 - 1)
            # abd_tensor = abd_tensor + perturbation
        else:
            psfmap = restore_psfmap_from_table(abd_tensor).permute(2,0,1)
            # print(psfmap.shape)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'abd': psfmap,  'fov_mask':  weights_expanded, 'fov': fov, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)