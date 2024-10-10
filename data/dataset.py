r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader
from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS


# --------------------------
# 無資料擴增
# ---------------------------
class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'fss': DatasetFSS,
            'coco': DatasetCOCO,
            'pascal': DatasetPASCAL,
        }

        # FSS
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std  = [0.229, 0.224, 0.225]


        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor()])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader


# --------------------------
# 資料擴增
# --------------------------
# class FSSDataset:

#     @classmethod
#     def initialize(cls, img_size, datapath, use_original_imgsize, augment=False):

#         cls.datasets = {
#             'fss': DatasetFSS,
#             'coco': DatasetCOCO,
#             'pascal': DatasetPASCAL,
#         }

#         cls.img_size = img_size  # 添加這一行以設置 img_size
#         cls.img_mean = [0.485, 0.456, 0.406]
#         cls.img_std  = [0.229, 0.224, 0.225]
#         cls.datapath = datapath
#         cls.use_original_imgsize = use_original_imgsize

#         if augment:
#             cls.transform = transforms.Compose([
#                 transforms.Resize(size=(img_size, img_size)),
#                 transforms.RandomResizedCrop(img_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=cls.img_mean, std=cls.img_std),
#             ])
#         else:
#             cls.transform = transforms.Compose([
#                 transforms.Resize(size=(img_size, img_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=cls.img_mean, std=cls.img_std),
#             ])

#     @classmethod
#     def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, augment=False):
#         shuffle = split == 'trn'
#         nworker = nworker if split == 'trn' else 0

#         # Initialize with or without augmentation based on split
#         cls.initialize(cls.img_size, cls.datapath, cls.use_original_imgsize, augment=augment)
        
#         dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
#         dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

#         return dataloader

