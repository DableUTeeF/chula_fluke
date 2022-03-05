from mmtrain import train
import data
import models
import os


if __name__ == '__main__':
    cfg = models.RetinaNet.rsb_50()
    cfg = data.add_val_data(cfg)
    cfg.checkpoint_config.max_keep_ckpts = 1
    cfg.load_from = '/media/palm/BiggerData/Chula_Parasite/checkpoints/retinanet_rsb50/start/epoch_20.pth'
    # cfg.resume_from = '/media/palm/BiggerData/Chula_Parasite/checkpoints/tood_r50/ft/epoch_3.pth'
    cfg.work_dir = '/media/palm/BiggerData/Chula_Parasite/checkpoints/retinanet_rsb50/ft'
    train(cfg, 100)

