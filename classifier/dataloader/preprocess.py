from torchvision import transforms
from PIL import Image
import PIL
from .randaugment import RandAugment
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def bboxaugprocess(size, advprop):
    if advprop:
        nomalize = A.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        nomalize = A.Normalize()

    albumentations_transform = A.Compose(
        [
            A.LongestMaxSize(size, interpolation = cv2.INTER_AREA),
            A.PadIfNeeded(size, size, value=(255,255,255), border_mode=0),
            A.RandomResizedCrop(size,size, p=1, interpolation = cv2.INTER_AREA),
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
            ], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
            nomalize
        ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    )

    return albumentations_transform


def bboxvalprocess(size, advprop):

    if advprop:
        nomalize = A.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        nomalize = A.Normalize()

    albumentations_transform = A.Compose([
        A.LongestMaxSize(size, interpolation = cv2.INTER_AREA),
        A.PadIfNeeded(size, size, value=(255,255,255), border_mode=0),
        A.Resize(size,size, interpolation = cv2.INTER_AREA),
        nomalize
    ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    )
    return albumentations_transform


def bboxtestprocess(size, advprop):

    if advprop:
        nomalize = A.Lambda(lambda image: image * 2.0 - 1.0)
    else:
        nomalize = A.Normalize()

    albumentations_transform = A.Compose([
        A.LongestMaxSize(size, interpolation = cv2.INTER_AREA),
        A.PadIfNeeded(size, size, value=(255,255,255), border_mode=0),
        A.Resize(size,size, interpolation = cv2.INTER_AREA),
        nomalize
    ]
    )
    return albumentations_transform

class advprop_normal(A.ImageOnlyTransform):
    def __init__(self, a=2.0, b=1.0, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)
        self.a = a
        self.b = b

    def apply(self, img, **params):
        return img.astype(np.float32) * 2.0 - 1.0

    def get_transform_init_args_names(self):
        return ('a', 'b')

def clsalbtrain(size, advprop, size_fixing = False):
    if advprop:
        nomalize = advprop_normal()
    else:
        nomalize = A.Normalize()

    if size_fixing :
        albumentations_transform = A.Compose(
            [
                A.SmallestMaxSize(size, interpolation = cv2.INTER_AREA),
                A.CenterCrop(size,size, p=1),
                A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ], p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                nomalize,
                ToTensorV2()
            ]
        )

    else:
        albumentations_transform = A.Compose(
            [
                A.SmallestMaxSize(size, interpolation = cv2.INTER_AREA),
                A.RandomResizedCrop(size,size, scale=(0.6, 1.0), interpolation = cv2.INTER_AREA),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(p=0.5),
                    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.5),
                    A.GaussNoise(p=0.5),
                    A.JpegCompression(quality_lower=0, quality_upper=100, p=0.5),
                ], p=0.2),
                A.OneOf([
                    A.MotionBlur(p=5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=(3,20), p=0.5),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=0.5),
                    A.IAAPiecewiseAffine(p=0.5),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.ChannelShuffle(p=0.5),
                    A.IAASharpen(p=0.5),
                    A.IAAEmboss(p=0.5),
                    A.ToSepia(p=0.5),
                    A.ToGray(p=0.5),
                    A.InvertImg(p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ], p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.CoarseDropout(p=0.5),
                    A.GridDropout(p=0.5),
                    A.RandomGridShuffle(grid=(3, 3), p=0.5)
                ], p=0.2),
                nomalize,
                ToTensorV2(p=1)
            ]
        )

    return albumentations_transform

def clsalbval(size, advprop):
    if advprop:
        nomalize = advprop_normal()
    else:
        nomalize = A.Normalize()

    albumentations_transform = A.Compose([
        A.SmallestMaxSize(size, interpolation = cv2.INTER_AREA),
        A.CenterCrop(size,size, p=1),
        nomalize,
        ToTensorV2()
    ])
    return albumentations_transform

def albinversepreprocess(advprop):
    if advprop:
        nomalize = A.Lambda(lambda img: (img + 1.0)*(1/2))
    else:
        nomalize = A.Normalize(mean = [-0.485/0.229,-0.456/0.224,-0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225])
    
    return A.Compose([nomalize])
    
def randaugprocess(size, advprop, no_cutout = False, resize_op = False, n=2, m=10):
    if advprop:
        nomalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        nomalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if resize_op :
        resize = transforms.RandomResizedCrop((size,size))
    else:
        resize = transforms.RandomResizedCrop(size)
    preprocess = transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(),
        RandAugment(n,m),
        transforms.ToTensor(),
        nomalize
    ])
    return preprocess

def randaugafterprocess(size, advprop, no_cutout = False, resize_op = False, n=2, m=10):
    if advprop:
        nomalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        nomalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if resize_op :
        resize = transforms.Resize((size,size))
    else:
        resize = transforms.Resize(size)

    preprocess = transforms.Compose([
        resize,
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        #RandAugment(n,m),
        transforms.ToTensor(),
        nomalize
    ])
    return preprocess

def valpreprocess(size, advprop, resize_op = False):
    if advprop:
        nomalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        nomalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if resize_op :
        resize = transforms.Resize((size,size))
    else:
        resize = transforms.Resize(size)

    preprocess = transforms.Compose([
        resize,
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        nomalize
    ])
    return preprocess

def inversepreprocess(advprop):

    if advprop:
        nomalize = transforms.Lambda(lambda img: (img + 1.0)*(1/2))
    else:
        nomalize = transforms.Normalize(mean = [-0.485/0.229,-0.456/0.224,-0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225])

    inversetransform = transforms.Compose([
        nomalize
    ])
    return inversetransform