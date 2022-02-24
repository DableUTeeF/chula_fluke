import os.path as osp
from mmdet.apis import set_random_seed

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset, COCO
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os
from PIL import Image


@DATASETS.register_module()
class FlukeDataset(CocoDataset):
    CLASSES = (
        'Ascaris lumbricoides',
        'Capillaria philippinensis',
        'Enterobius vermicularis',
        'Fasciolopsis buski',
        'Hookworm egg',
        'Hymenolepis diminuta',
        'Hymenolepis nana',
        'Opisthorchis viverrine',
        'Paragonimus spp',
        'Taenia spp. egg',
        'Trichuris trichiura'
    )

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = list(range(len(self.CLASSES)))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            # print(info)
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos


if __name__ == '__main__':
    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py')
    cfg.model.bbox_head.num_classes = 11

    cfg.dataset_type = 'FlukeDataset'
    cfg.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data'

    cfg.data.train.type = 'FlukeDataset'
    cfg.data.train.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.data.train.ann_file = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/labels.json'
    cfg.data.train.img_prefix = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.work_dir = '/media/palm/BiggerData/Chula_Parasite/checkpoints/vfnet_r50'

    # cfg.load_from = '/media/palm/BiggerData/mmdetection/cp/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'
    cfg.resume_from = '/media/palm/BiggerData/Chula_Parasite/checkpoints/vfnet_r50/epoch_23.pth'

    cfg.optimizer.lr = cfg.optimizer.lr / 4
    cfg.log_config.interval = 100

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [0]

    cfg.checkpoint_config.interval = 1
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)
