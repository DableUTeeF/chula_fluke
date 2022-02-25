import os.path as osp
from mmdet.apis import set_random_seed

import mmcv
import models

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


@DATASETS.register_module()
class FlukeSplitDataset(CocoDataset):
    CLASSES = ('Ascaris lumbricoides',
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
        ann_file, split, seed = ann_file.split('|')
        seed = int(seed)
        self.coco = COCO(ann_file)
        self.cat_ids = list(range(len(self.CLASSES)))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for idx, i in enumerate(self.img_ids):
            if split == 'train' and idx % 10 == seed:
                continue
            elif split == 'val' and idx % 10 != seed:
                continue
            info = self.coco.load_imgs([i])[0]
            # print(info)
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos


class Transform:
    @staticmethod
    def albu1():
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        albu_train_transforms = [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=30,
                interpolation=2,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0)
                ],
                p=0.1),
            dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            # dict(
            #     type='Resize',
            #     img_scale=[(1400, 600), (1400, 1024)],
            #     multiscale_mode='range',
            #     keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Albu',
                transforms=albu_train_transforms,
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap={
                    'img': 'image',
                    'gt_bboxes': 'bboxes'
                },
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        return train_pipeline


def add_train_data(cfg):
    cfg.dataset_type = 'FlukeSplitDataset'
    cfg.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data'

    cfg.data.train.type = 'FlukeSplitDataset'
    cfg.data.train.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.data.train.ann_file = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/labels.json|train|0'
    cfg.data.train.img_prefix = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.work_dir = '/media/palm/BiggerData/Chula_Parasite/checkpoints/cascade_r101_albu_finetunded/start'
    return cfg


def add_val_data(cfg):
    cfg.dataset_type = 'FlukeSplitDataset'
    cfg.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data'

    cfg.data.train.type = 'FlukeSplitDataset'
    cfg.data.train.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.data.train.ann_file = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/labels.json|val|0'
    cfg.data.train.img_prefix = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.work_dir = '/media/palm/BiggerData/Chula_Parasite/checkpoints/cascade_r101_albu_finetunded/start'
    return cfg


def add_all_data(cfg):
    cfg.dataset_type = 'FlukeDataset'
    cfg.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data'

    cfg.data.train.type = 'FlukeSplitDataset'
    cfg.data.train.data_root = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.data.train.ann_file = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/labels.json'
    cfg.data.train.img_prefix = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/data/'
    cfg.work_dir = '/media/palm/BiggerData/Chula_Parasite/checkpoints/cascade_r101_albu_finetunded/start'
    return cfg


def train(
        cfg,
        lr_divide,
        checkpoint_config_interval=4,

):
    cfg.optimizer.lr = cfg.optimizer.lr / lr_divide
    cfg.log_config.interval = 100

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [0]

    cfg.checkpoint_config.interval = checkpoint_config_interval
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)


def train_std(
        config_method,
        working_dir,
        resume_from=None,
        load_from=None,
):
    cfg = config_method()
    cfg = add_all_data(cfg)
    cfg.work_dir = working_dir
    if resume_from:
        cfg.resume_from = resume_from
    elif load_from:
        cfg.load_from = load_from
    train(cfg, 4)


def train_ft(
        config_method,
        working_dir,
        resume_from=None,
        load_from=None,
):
    cfg = config_method()
    cfg = add_train_data(cfg)
    max_epochs = cfg.runner.max_epochs
    if resume_from:
        cfg.resume_from = resume_from
    elif load_from:
        cfg.load_from = load_from
    cfg.work_dir = os.path.join(working_dir, 'start')
    train(cfg, 4)

    wd = cfg.work_dir
    cfg = config_method()
    cfg = add_val_data(cfg)
    cfg.load_from = os.path.join(wd, f'epoch_{max_epochs}.pth')
    cfg.work_dir = os.path.join(working_dir, 'ft')
    train(cfg, 100, checkpoint_config_interval=int(max_epochs / 2))


if __name__ == '__main__':
    train_std(models.RetinaNet.resnet_50(),
              '/media/palm/BiggerData/Chula_Parasite/checkpoints/retinanet_r50_std',
              load_from='/media/palm/BiggerData/mmdetection/cp/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth')
    train_ft(models.RetinaNet.resnet_50(),
             '/media/palm/BiggerData/Chula_Parasite/checkpoints/retinanet_r50_ft',
             load_from='/media/palm/BiggerData/mmdetection/cp/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth')
