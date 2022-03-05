import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
from mmdet.apis import set_random_seed
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import models

if __name__ == '__main__':
    cfg = models.CascadeRCNN.pvtv2_b3()

    cfg.data.samples_per_gpu = 2
    cfg.work_dir = 'checkpoints/cascade_pvt-v2-b3'

    cfg.data_root = 'data/'  # todo: change to coco folder
    cfg.data.train.img_prefix = 'data/images/train2017/'
    cfg.data.train.ann_file = 'data/annotations/instances_train2017.json',

    cfg.optimizer = dict(
        # _delete_=True,
        type='AdamW', lr=0.00005, weight_decay=0.0001)

    cfg.log_config.interval = 1000
    cfg.checkpoint_config.max_keep_ckpts = 1

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [0]

    cfg.checkpoint_config.interval = 1
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)
