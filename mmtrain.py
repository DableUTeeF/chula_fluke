import os.path as osp
import models
from mmdet.apis import set_random_seed

import mmcv
import data
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os
import schedules

def train(
        cfg,
        lr_divide,
        checkpoint_config_interval=1,
        log_interval=100,

):
    cfg.optimizer.lr = cfg.optimizer.lr / lr_divide
    cfg.log_config.interval = log_interval
    cfg = schedules.LinearSchedules.set_20e(cfg)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

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
    cfg.checkpoint_config.max_keep_ckpts = 1
    cfg = data.add_all_data(cfg)
    cfg.work_dir = os.path.join(working_dir, 'std')
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
        max_epochs=20,
        transforms=None
):
    cfg = config_method()
    cfg.checkpoint_config.max_keep_ckpts = 1
    cfg = data.add_train_data(cfg)
    if transforms:
        cfg = data.Transform.yolo_mosaic(cfg)
    cfg.data.workers_per_gpu = 8
    if resume_from:
        cfg.resume_from = resume_from
    elif load_from:
        cfg.load_from = load_from
    cfg.work_dir = os.path.join(working_dir, 'start')
    train(cfg, 4)

    wd = cfg.work_dir
    cfg = config_method()
    cfg = data.add_val_data(cfg)
    if transforms:
        cfg = data.Transform.yolo_mosaic(cfg)
    cfg.checkpoint_config.max_keep_ckpts = 1
    cfg.data.workers_per_gpu = 8
    cfg.load_from = os.path.join(wd, f'epoch_{max_epochs}.pth')
    cfg.work_dir = os.path.join(working_dir, 'ft')
    train(cfg, 100)


if __name__ == '__main__':
    # todo: the high score 101 didn't init new model
    # train_std(models.CascadeRCNN.resnet_50,
    #           '/media/palm/BiggerData/Chula_Parasite/checkpoints/cascade_r50',
    #           load_from='/media/palm/BiggerData/mmdetection/cp/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth',
    #           # resume_from='/media/palm/BiggerData/Chula_Parasite/checkpoints/yolo_v3/std/epoch_18.pth'
    #           )
    # train_ft(models.Yolo.generic,
    #          '/media/palm/BiggerData/Chula_Parasite/checkpoints/yolo_mosaic',
    #          load_from='/media/palm/BiggerData/mmdetection/cp/yolov3_d53_mstrain-608_273e_coco-139f5633.pth',
    #          # resume_from='/media/palm/BiggerData/Chula_Parasite/checkpoints/retinanet_rsb50/start/epoch_1.pth'
    #          transforms=True
    #          )
    train_ft(models.VFNet.resnet_50,
             '/media/palm/BiggerData/Chula_Parasite/checkpoints/vfnet_r50',
             # load_from='/media/palm/BiggerData/mmdetection/cp/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth',
             resume_from='/media/palm/BiggerData/Chula_Parasite/checkpoints/vfnet_r50/start/epoch_4.pth'
             )
