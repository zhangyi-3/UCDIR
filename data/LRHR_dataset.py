import os.path
from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util

import io
import cv2
import traceback, torch
import numpy as np

from data.mask import bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox

try:
    import mc
except:
    print('** import mc error')
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)

    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


class MemcachedBase(Dataset):
    def __init__(self):
        super(MemcachedBase, self).__init__()
        self.initialized = False

        import socket
        hostname = socket.gethostname()
        if '-142-' in hostname:  # disable petrel
            from petrel_client.client import Client
            conf_path = '~/petreloss.conf'
            client = Client(conf_path)  # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
        else:
            client = None
            print('\rimport client error', end='')

        self.ceph = client  # self.ceph = None  # disable ceph

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/zhangyi2.vendor/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _mem_load(self, filename, flag=''):
        if self.ceph is not None:
            filename = self.ceph_rename(filename)
            return self.ceph_load(filename, flag, self.ceph)

        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)

        if flag == 'gray':
            img_array = np.frombuffer(value_str, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=2)  # print('gray', img.shape)
        else:
            img = pil_loader(value_str)
        return img

    def ceph_rename(self, path):
        if 's3:' == path[:3]:
            return path
        if 'share/images/' in path:
            shortpath = path.split('share/images/')[-1]
            img_url = 's3://zy3_imagenet/' + shortpath
            shortpath = path.split('share/')[-1]
            img_url = 's3://zhangyi/' + shortpath
        elif '.vendor/' in path:
            shortpath = path.split('.vendor/')[-1]
            img_url = 's3://zhangyi/' + shortpath
        elif 'zhangyi3/Restormer/' in path:
            shortpath = path.split('zhangyi3/Restormer/')[-1]
            img_url = 's3://Restormer/' + shortpath
        else:
            assert NotImplementedError
        return img_url

    def ceph_load(self, path, flag, client):
        img_url = path
        # 图片读取
        img_bytes = client.get(img_url)
        assert (img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        if flag == 'gray':
            img_array = np.frombuffer(img_mem_view, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=2)
        else:
            img = pil_loader(
                img_mem_view)  # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # img = np.array(img)[..., ::-1]
        return img

    def get_stream(self, path):
        url = self.ceph_rename(path)
        # print(url, path, 'get')
        value = self.ceph.get(url)
        value_buf = io.BytesIO(value)
        value_buf.seek(0)
        return value_buf

    def put_stream(self, path, obj):  # buffer = io.BytesIO()
        url = self.ceph_rename(path)
        # print(url, path, 'put')
        obj.seek(0)
        self.ceph.put(url, obj)
        return

    def torchsave(self, state_dict, path):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        self.put_stream(path, buffer)
        return

    def torchload(self, path, map_location='cpu'):
        test = self.get_stream(path)
        test = torch.load(test, map_location=map_location)
        return test


class LRHRDataset(MemcachedBase):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1,
                 need_LR=False):
        super(LRHRDataset, self).__init__()
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype in ['img', 'mc']:

            # support automatic ceph
            if self.ceph is not None:
                sr_path = '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution)
                hr_path = '{}/hr_{}'.format(dataroot, r_resolution)
                self.sr_path = list(self.ceph.list(self.ceph_rename(sr_path)))
                self.hr_path = list(self.ceph.list(self.ceph_rename(hr_path)))

                self.sr_path = [os.path.join(self.ceph_rename(sr_path), item) for item in self.sr_path]
                self.hr_path = [os.path.join(self.ceph_rename(hr_path), item) for item in self.hr_path]
                print('\r** ceph start **', len(self.sr_path), self.sr_path[0])

                if self.need_LR:
                    lr_path = '{}/lr_{}'.format(dataroot, l_resolution)
                    self.lr_path = list(self.ceph.list(self.ceph_rename(lr_path)))
                    self.lr_path = [os.path.join(self.ceph_rename(lr_path), item) for item in self.lr_path]
            else:
                self.sr_path = Util.get_paths_from_images('{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
                self.hr_path = Util.get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))

                if self.need_LR:
                    self.lr_path = Util.get_paths_from_images('{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get('hr_{}_{}'.format(self.r_res, str(index).zfill(5)).encode('utf-8'))
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8'))
                if self.need_LR:
                    lr_img_bytes = txn.get('lr_{}_{}'.format(self.l_res, str(index).zfill(5)).encode('utf-8'))
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len - 1)
                    hr_img_bytes = txn.get('hr_{}_{}'.format(self.r_res, str(new_index).zfill(5)).encode('utf-8'))
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8'))
                    if self.need_LR:
                        lr_img_bytes = txn.get('lr_{}_{}'.format(self.l_res, str(new_index).zfill(5)).encode('utf-8'))
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'mc':
            img_HR = self._mem_load(self.hr_path[index])
            img_SR = self._mem_load(self.sr_path[index])
            if self.need_LR:
                img_LR = self._mem_load(self.lr_path[index])
            assert img_SR is not None and img_HR is not None  # print(img_HR.shape, img_SR.shape, np.mean(img_HR))
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment([img_LR, img_SR, img_HR], split=self.split,
                min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}


class PairDataset(MemcachedBase):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, **kwargs):
        super(PairDataset, self).__init__()
        self.datatype = datatype

        self.data_len = data_len
        # self.need_LR = need_LR
        self.split = split
        self.crop_size = kwargs.get('crop_size', 0)
        self.mask = kwargs.get('mask', -1)

        if datatype in ['img', 'mc']:
            sr_path, hr_path = dataroot['lq'], dataroot['gt']
            # support automatic ceph
            if self.ceph is not None and datatype == 'mc':
                self.sr_path = list(self.ceph.list(self.ceph_rename(sr_path)))
                self.hr_path = list(self.ceph.list(self.ceph_rename(hr_path)))

                self.sr_path = [os.path.join(self.ceph_rename(sr_path), item) for item in self.sr_path]
                self.hr_path = [os.path.join(self.ceph_rename(hr_path), item) for item in self.hr_path]
                print('** ceph start **', len(self.sr_path), self.sr_path[0], end=' ')

            else:
                self.sr_path = Util.get_paths_from_images(sr_path)
                self.hr_path = Util.get_paths_from_images(hr_path)

                # if self.need_LR:  #     self.lr_path = Util.get_paths_from_images(  #         '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))  # print('datalen', self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        while True:
            try:
                if self.datatype == 'mc':
                    img_HR = self._mem_load(self.hr_path[index])
                    img_SR = self._mem_load(self.sr_path[index])
                    # if self.need_LR:
                    #     img_LR = self._mem_load(self.lr_path[index])
                    assert img_SR is not None and img_HR is not None  # print(img_HR.shape, img_SR.shape, np.mean(img_HR))
                else:
                    img_HR = Image.open(self.hr_path[index]).convert("RGB")
                    img_SR = Image.open(self.sr_path[index]).convert(
                        "RGB")  # if self.need_LR:  #     img_LR = Image.open(self.lr_path[index]).convert("RGB")
                if self.crop_size > 0:
                    h, w = img_HR.size  # for pillow image
                    hs, ws = np.random.randint(h - self.crop_size), np.random.randint(w - self.crop_size)
                    img_HR = img_HR.crop((hs, ws, hs + self.crop_size, ws + self.crop_size))
                    img_SR = img_SR.crop((hs, ws, hs + self.crop_size, ws + self.crop_size))

                [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))

                mask = 0
                if self.mask > 0:
                    self.mask_mode = 'free_form'
                    self.image_size = (img_SR.shape[1], img_SR.shape[2])
                    mask = self.get_mask()  # uint8
                # print('make',img_SR.shape, mask.shape)
                # print(mask.float().mean(), mask.min(), mask.max())
                return {'HR': img_HR, 'SR': img_SR, 'Index': index, 'mask': mask}
            except:
                print('**error', self.hr_path[index])
                print(traceback.format_exc())
                index = np.random.randint(self.data_len)
                continue

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class SingleDataset(MemcachedBase):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, **kwargs):
        super(SingleDataset, self).__init__()
        self.datatype = datatype

        self.data_len = data_len
        # self.need_LR = need_LR
        self.split = split
        self.crop_size = kwargs.get('crop_size', 0)

        if datatype in ['img', 'mc']:
            hr_path = dataroot['gt']
            # support automatic ceph
            if self.ceph is not None:
                self.hr_path = list(self.ceph.list(self.ceph_rename(hr_path)))

                self.hr_path = [os.path.join(self.ceph_rename(hr_path), item) for item in self.hr_path]
                print('** ceph start **', len(self.hr_path), self.hr_path[0])

            else:
                self.hr_path = Util.get_paths_from_images(hr_path)

                # if self.need_LR:  #     self.lr_path = Util.get_paths_from_images(  #         '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(datatype))
        print('datalen', self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        while True:
            try:
                if self.datatype == 'mc':
                    img_HR = self._mem_load(self.hr_path[
                                                index])  # if self.need_LR:  #     img_LR = self._mem_load(self.lr_path[index])  # print(img_HR.shape, img_SR.shape, np.mean(img_HR))
                else:
                    img_HR = Image.open(self.hr_path[index]).convert(
                        "RGB")  # if self.need_LR:  #     img_LR = Image.open(self.lr_path[index]).convert("RGB")
                if self.crop_size > 0:
                    h, w = img_HR.size  # for pillow image
                    hs, ws = np.random.randint(h - self.crop_size), np.random.randint(w - self.crop_size)
                    img_HR = img_HR.crop((hs, ws, hs + self.crop_size, ws + self.crop_size))

                [img_HR] = Util.transform_augment([img_HR], split=self.split, min_max=(-1, 1))
                return {'HR': img_HR, 'Index': index}
            except:
                print('**error', self.hr_path[index])
                index = np.random.randint(self.data_len)
                continue


from torchvision.transforms import functional as trans_fn


class ImagenetSRDataset(MemcachedBase):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, **kwargs):
        super(ImagenetSRDataset, self).__init__()
        self.datatype = datatype

        self.data_len = data_len
        # self.need_LR = need_LR
        self.split = split
        self.sizes = [64, 256]

        if datatype in ['img', 'mc']:
            self.root = dataroot['root']
            # self.hr_path = np.loadtxt(dataroot['txt'], dtype=np.str)[:, 0]  # sometime JPEG -> JPE
            with open(dataroot['txt'], 'r') as f:
                res = f.readlines()
            self.hr_path = [item.split(' ')[0] for item in res]

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(datatype))
        print('imagenet sr datalen', self.data_len)
        self.sr_path = self.hr_path  # for testing

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        while True:
            try:
                tp_path = os.path.join(self.root, self.hr_path[index])
                if self.datatype == 'mc':
                    img = self._mem_load(tp_path)
                else:
                    img = Image.open(tp_path).convert("RGB")

                resample = Image.BICUBIC
                #    lr_img = resize_and_convert(img, self.sizes[0], resample)
                #    img_HR = resize_and_convert(img, self.sizes[1], resample)
                #    img_SR = resize_and_convert(lr_img, self.sizes[1], resample)
                if min(img.size) < self.sizes[-1]:
                    img = trans_fn.resize(img, self.sizes[-1], resample)

                img_HR = trans_fn.center_crop(img, min(img.size))
                img_HR = trans_fn.resize(img_HR, self.sizes[-1], resample)
                img_LR = trans_fn.resize(img_HR, self.sizes[0], resample)
                img_SR = trans_fn.resize(img_LR, self.sizes[-1], resample)

                [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))

                return {'HR': img_HR, 'SR': img_SR, 'Index': index, }
            except:
                print('**error', self.hr_path[index])
                print(traceback.format_exc())
                index = np.random.randint(self.data_len)
                continue


class ImagenetJPGDataset(MemcachedBase):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, **kwargs):
        super(ImagenetJPGDataset, self).__init__()
        self.datatype = datatype

        self.data_len = data_len
        # self.need_LR = need_LR
        self.split = split
        self.crop_size = kwargs.get('crop_size', 0)
        self.factor = kwargs.get('factor', [5, 5])

        if datatype in ['img', 'mc']:
            self.root = dataroot['root']
            # self.hr_path = np.loadtxt(dataroot['txt'], dtype=np.str)[:, 0]  # sometime JPEG -> JPE
            with open(dataroot['txt'], 'r') as f:
                res = f.readlines()
            self.hr_path = [item.split(' ')[0] for item in res]

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(datatype))
        print('imagenet sr datalen', self.data_len)
        self.sr_path = self.hr_path  # for testing

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        while True:
            try:
                tp_path = os.path.join(self.root, self.hr_path[index])
                if self.datatype == 'mc':
                    img = self._mem_load(tp_path)
                else:
                    img = Image.open(tp_path).convert("RGB")

                img_HR = img
                if min(img_HR.size) < self.crop_size:
                    img_HR = img_HR.resize((self.crop_size, self.crop_size))

                if self.crop_size > 0:
                    h, w = img_HR.size  # for pillow image
                    hs, ws = (h - self.crop_size) // 2, (w - self.crop_size) // 2
                    img_HR = img_HR.crop((hs, ws, hs + self.crop_size, ws + self.crop_size))
                else:  # crop to proper size
                    h, w = img_HR.size  # for pillow image
                    factor = 16
                    tp = [h // factor * factor, w // factor * factor]
                    hs, ws = (h - tp[0]) // 2, (w - tp[1]) // 2
                    img_HR = img_HR.crop((hs, ws, hs + tp[0], ws + tp[1]))

                # compress
                img_SR = np.array(img_HR)
                quality_factor = self.factor[0] if self.factor[0] == self.factor[1] else np.random.randint(
                    self.factor[0], self.factor[1] + 1)
                _, encimg = cv2.imencode('.jpg', img_SR, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_SR = cv2.imdecode(encimg, 3)

                [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))

                return {'HR': img_HR, 'SR': img_SR, 'Index': index, }
            except:
                print('**error', self.hr_path[index])
                print(traceback.format_exc())
                index = np.random.randint(self.data_len)
                continue


class ImagenetColorDataset(ImagenetJPGDataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, **kwargs):
        super(ImagenetColorDataset, self).__init__(dataroot, datatype, split, data_len, kwargs)

    def __getitem__(self, index):
        while True:
            try:
                tp_path = os.path.join(self.root, self.hr_path[index])
                if self.datatype == 'mc':
                    img = self._mem_load(tp_path)
                else:
                    img = Image.open(tp_path).convert("RGB")

                img_HR = img
                if min(img_HR.size) < self.crop_size:
                    img_HR = img_HR.resize((self.crop_size, self.crop_size))

                if self.crop_size > 0:
                    h, w = img_HR.size  # for pillow image
                    hs, ws = (h - self.crop_size) // 2, (w - self.crop_size) // 2
                    img_HR = img_HR.crop((hs, ws, hs + self.crop_size, ws + self.crop_size))
                else:  # crop to proper size
                    h, w = img_HR.size  # for pillow image
                    factor = 16
                    tp = [h // factor * factor, w // factor * factor]
                    hs, ws = (h - tp[0]) // 2, (w - tp[1]) // 2
                    img_HR = img_HR.crop((hs, ws, hs + tp[0], ws + tp[1]))

                # compress tp gray
                img_SR = np.array(img_HR)
                img_SR = img_SR.mean(axis=-1, keepdims=True)
                img_SR = np.concatenate([img_SR] * 3, axis=-1)

                [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))

                return {'HR': img_HR, 'SR': img_SR, 'Index': index, }
            except:
                print('**error', self.hr_path[index])
                print(traceback.format_exc())
                index = np.random.randint(self.data_len)
                continue


import math, time


def imfrombytes(content, flag='color', float32=False):
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


from data.degradations import circular_lowpass_kernel, random_mixed_kernels

param = {  # real-esrgan
    'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], 'sinc_prob': 0.1, 'blur_sigma': [0.2, 3],
    'betag_range': [0.5, 4], 'betap_range': [1, 2],

    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], 'sinc_prob2': 0.1, 'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4], 'betap_range2': [1, 2],

    'final_sinc_prob': 0.8,

    'use_hflip': True, 'use_rot': False, }
param1 = {'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], 'sinc_prob': 0.1, 'blur_sigma': [0.2, 1.5],
    'betag_range': [0.5, 2.0], 'betap_range': [1, 1.5],

    'blur_kernel_size2': 11,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], 'sinc_prob2': 0.1, 'blur_sigma2': [0.2, 1.0],
    'betag_range2': [0.5, 2.0], 'betap_range2': [1, 1.5],

    'final_sinc_prob': 0.8,

    # gt_size: 512,
    'use_hflip': True, 'use_rot': False, }


class RealESRGANDataset(ImagenetJPGDataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, **kwargs):
        super(RealESRGANDataset, self).__init__(dataroot, datatype, split, data_len, **kwargs)

        opt = eval(kwargs.get('param', 'param'))

        self.opt = opt
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        # gt_path = self.paths[index]
        gt_path = os.path.join(self.root, self.hr_path[index])
        # print('**', gt_path, self.root, self.ceph_rename(gt_path))
        # exit()
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                # img_bytes = self.file_client.get(gt_path, 'gt')
                tp_path = gt_path
                if self.datatype == 'mc':
                    img = self._mem_load(tp_path)
                else:
                    img = Image.open(tp_path).convert("RGB")

                img_gt = np.array(img).astype(np.float32) / 255.
                img_gt = img_gt[..., ::-1]  # fit bgr format here.

            except (IOError, OSError) as e:
                # change another file to read
                print('*error', gt_path)
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        # img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        if self.split == 'train':
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        elif self.split == 'val':
            pass
        else:
            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            return {'lq': img_gt, 'gt': img_gt, 'Index': index}

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, kernel_size, self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi], self.betag_range, self.betap_range, noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(self.kernel_list2, self.kernel_prob2, kernel_size, self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi], self.betag_range2, self.betap_range2, noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d


if __name__ == '__main__':
    print('test')
    path = '../../../Downloads/zhangyi3_circle.jpg'
    img = Image.open(path).convert("RGB")
    # img = img.resize((64, 64))
    print(img.size)

    img = img.crop((0, 0, 64, 64))
    img_SR = np.array(img)
    quality_factor = 5
    _, encimg = cv2.imencode('.jpg', img_SR, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img_SR = cv2.imdecode(encimg, 3)

    img_L = np.array(img)
    quality_factor = 5
    _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img_L = cv2.imdecode(encimg, 3)

    print(img_L.shape)

    cv2.imwrite(path + '.jpg', img_L[..., ::-1])
