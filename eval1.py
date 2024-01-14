import csv
import cv2, os
import torch
import argparse
import lpips

from torch import nn
import numpy as np

from scipy.stats import entropy
from cleanfid import fid
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from metric.niqe import calculate_niqe
from metric.ssim import calculate_ssim
from PIL import Image

from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as data
from torchvision import transforms
import torch.utils.data

from torchvision.models.inception import inception_v3

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


class BaseDataset(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], loader=pil_loader):
        self.imgs = make_dataset(data_root)
        self.tfs = transforms.Compose([transforms.Resize((image_size[0], image_size[1])), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        return img

    def __len__(self):
        return len(self.imgs)


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class LPIPS:
    def __init__(self, net='alex'):
        """
        args:
            net: alex/vgg/squeeze
            if_spatial: return a score (False) or a map of scores (True).
        """
        self.net = net
        self.if_spatial = False

        self.lpips_fn = lpips.LPIPS(net=self.net, spatial=self.if_spatial, verbose=False)

        self.if_cuda = True if torch.cuda.is_available() else False
        if self.if_cuda:
            self.lpips_fn.cuda()

    def _preprocess(self, img):
        img = img[:, :, ::-1]  # (H W BGR) -> (H W RGB)
        img = img / (255. / 2.) - 1.  # -> [0, 2] -> [-1, 1]
        img = img.transpose(2, 0, 1)  # ([RGB] H W)
        out = torch.Tensor(img)
        out = torch.unsqueeze(out, 0)  # (1 [RGB] H W)
        if self.if_cuda:
            out = out.cuda()
        return out

    def forward(self, img1, img2):
        """
        input:
            img1/img2: (H W C) uint8 ndarray.
        return:
            lpips score, float.
        """
        img1, img2 = img1.copy(), img2.copy()
        img1, img2 = self._preprocess(img1), self._preprocess(img2)
        lpips_score = self.lpips_fn.forward(img1, img2)
        return lpips_score.item()


def save_csv(results, csv_path):
    results = list(results)
    with open(csv_path, 'a', newline='') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerows(results)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, default=None, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, default='results/vis/sids3-04', help='Generate images directory')
    parser.add_argument('-fid', type=float, default=0, help='Generate images directory')

    ''' parser configs '''
    args = parser.parse_args()
    RES = ["".join(args.dst.split('/')[::-1])]
    print('start')

    # LPIPS

    res = []
    gtdata = sorted([os.path.join(args.src, item) for item in os.listdir(args.src) if 'hr' in item])
    outdata = sorted([os.path.join(args.src, item) for item in os.listdir(args.src) if 'sr' in item])
    assert len(gtdata) == len(outdata)
    print(gtdata[0])
    print(outdata[0])
    # import pdb; pdb.set_trace()

    for idx, item in enumerate(gtdata):
        print('%2d/%d %s %s' % (idx, len(gtdata), item, outdata[idx]))
        head = item.split('_')[0]
        gt = cv2.imread(gtdata[idx])
        output = cv2.imread(outdata[idx])

        res.append([LPIPS('alex').forward(output, gt),
                    compare_psnr(output, gt, data_range=255),
                    calculate_ssim(output, gt, 0), calculate_niqe(output, 0, input_order='HWC', convert_to='y'),
                    calculate_niqe(output, 0, input_order='HWC', convert_to='y'),
                    ])

    res = np.array(res)
    print('LPIPS', np.mean(res[:, 0]))
    print('PSNR', np.mean(res[:, 1]))
    print('SSIM', np.mean(res[:, 2]))
    print('niqe', np.mean(res[:, 3]))  # print('brisque', np.mean(res[:, 4]))


# split to patches

    path = args.src
    if path[-1] == '/':
        path = path[:-1]
    gt_save_path = path + '_gt_pt'
    sr_save_path = path + '_sr_pt'

    for save_path, data in [(gt_save_path, gtdata),
                            (sr_save_path, outdata)]:
        os.makedirs(save_path, exist_ok=True)
        for _, item in enumerate(data):
            img = cv2.imread(item)
            img = np.array(img)
            print(_, item, img.shape)

            h, w, _ = img.shape
            ps = 256
            hs = h // ps * ps
            ws = w // ps * ps
            img = img[:hs, :ws]
            img = img.reshape(hs // ps, ps, ws // ps, ps, 3).swapaxes(1, 2).reshape(-1, ps, ps, 3)
            for idx, sub in enumerate(img):
                cv2.imwrite(os.path.join(save_path, os.path.basename(item)[:-4] + '%d.png' % idx), sub)

    # FID
    fid_score = fid.compute_fid(gt_save_path, sr_save_path, batch_size=32 * 8)
    kid_score = fid.compute_kid(gt_save_path, sr_save_path, batch_size=32 * 8)

    print('FID: {}'.format(fid_score))
    print('KID', kid_score)

