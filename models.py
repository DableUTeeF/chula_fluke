from mmcv import Config
import os
import schedules

main_path = '/media/palm/BiggerData/mmdetection/configs'


class Tood:
    @staticmethod
    def resnet_50():
        base_cfg = Config.fromfile(os.path.join(main_path, 'tood/tood_r50_fpn_1x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        return base_cfg


class GFL:
    @staticmethod
    def resnet_50():
        base_cfg = Config.fromfile(os.path.join(main_path, 'gfl/gfl_r50_fpn_1x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        return base_cfg


class VFNet:
    @staticmethod
    def resnet_50():
        base_cfg = Config.fromfile(os.path.join(main_path, 'vfnet/vfnet_r50_fpn_1x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        return base_cfg


class Faster_RCNN:
    @staticmethod
    def resnet(path):
        cfg = Config.fromfile(os.path.join(main_path, path))
        cfg.model.roi_head.bbox_head.num_classes = 11
        return cfg

    @staticmethod
    def resnet_50():
        return Faster_RCNN.resnet('faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

    @staticmethod
    def resnet_101():
        return Faster_RCNN.resnet('faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py')


class Yolo:
    @staticmethod
    def generic():
        cfg = Config.fromfile(os.path.join(main_path, 'yolo/yolov3_d53_mstrain-608_273e_coco.py'))
        cfg.model.bbox_head.num_classes = 11
        cfg.data.samples_per_gpu = 6
        return cfg


class RetinaNet:
    @staticmethod
    def rsb_50():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_r50_fpn_2x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
        model = dict(
            backbone=dict(
                init_cfg=dict(
                    type='Pretrained', prefix='backbone.', checkpoint=checkpoint)))
        base_cfg.model.backbone.init_cfg.type = 'Pretrained'
        base_cfg.model.backbone.init_cfg.prefix = 'backbone.'
        base_cfg.model.backbone.init_cfg.checkpoint = 'checkpoint'
        optimizer = dict(
            type='AdamW',
            lr=0.0001,
            weight_decay=0.05,
            paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
        base_cfg.optimizer = optimizer
        return base_cfg

    @staticmethod
    def resnet_50():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_r50_fpn_2x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        return base_cfg

    @staticmethod
    def resnet_101():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_r101_fpn_2x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        base_cfg = schedules.LinearSchedules.set_20e(base_cfg)
        base_cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
        return base_cfg

    @staticmethod
    def resnext_101_64():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_x101_64x4d_fpn_2x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        base_cfg = schedules.LinearSchedules.set_20e(base_cfg)
        return base_cfg

    @staticmethod
    def pvtv2_b0():
        base_cfg = Config.fromfile(os.path.join(main_path, 'pvt/retinanet_pvtv2-b0_fpn_1x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        base_cfg = schedules.LinearSchedules.set_20e(base_cfg)
        return base_cfg

    @staticmethod
    def pvtv2_b3():
        base_cfg = Config.fromfile(os.path.join(main_path, 'pvt/retinanet_pvtv2-b3_fpn_1x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        base_cfg = schedules.LinearSchedules.set_20e(base_cfg)
        return base_cfg

    @staticmethod
    def pvtv2_b1():
        base_cfg = Config.fromfile(os.path.join(main_path, 'pvt/retinanet_pvtv2-b1_fpn_1x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        base_cfg = schedules.LinearSchedules.set_20e(base_cfg)
        return base_cfg


class CascadeRCNN:
    @staticmethod
    def swin_s():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11
        cfg.model.backbone = dict(
            # _delete_=True,
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True)
        cfg.model.neck.in_channels = [96, 192, 384, 768]
        return cfg

    @staticmethod
    def swin_t():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11

        cfg.model.backbone = dict(
            # _delete_=True,
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True)
        cfg.model.neck.in_channels = [96, 192, 384, 768]
        return cfg

    @staticmethod
    def pvt_t():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11

        cfg.model.backbone = dict(
            # _delete_=True,
            type='PyramidVisionTransformer',
            num_layers=[2, 2, 2, 2])

        cfg.model.neck.in_channels = [64, 128, 320, 512]
        return cfg

    @staticmethod
    def pvt_s():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11

        cfg.model.backbone = dict(
            # _delete_=True,
            type='PyramidVisionTransformer',
            num_layers=[3, 4, 18, 3])

        cfg.model.neck.in_channels = [64, 128, 320, 512]
        return cfg

    @staticmethod
    def pvtv2_b0():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11

        cfg.model.backbone = dict(
            type='PyramidVisionTransformerV2',
            embed_dims=32,
            num_layers=[2, 2, 2, 2])

        cfg.model.neck.in_channels = [32, 64, 160, 256]
        return cfg

    @staticmethod
    def pvtv2_b3():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11

        cfg.model.backbone = dict(
            type='PyramidVisionTransformerV2',
            embed_dims=64,
            num_layers=[3, 4, 18, 3])

        cfg.model.neck.in_channels = [64, 128, 320, 512]
        return cfg

    @staticmethod
    def crpn():
        cfg = Config.fromfile(os.path.join(main_path, 'cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py'))
        cfg.model.roi_head.bbox_head.num_classes = 11
        return cfg

    @staticmethod
    def resnet_50():
        return CascadeRCNN.generic('cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py')

    @staticmethod
    def resnet_101():
        return CascadeRCNN.generic('cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py')

    @staticmethod
    def resnext_101_32():
        return CascadeRCNN.generic('cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco')

    @staticmethod
    def resnext_101_64():
        return CascadeRCNN.generic('cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco')

    @staticmethod
    def generic(path):
        cfg = Config.fromfile(os.path.join(main_path, path))
        cfg.model.roi_head.bbox_head[0].num_classes = 11
        cfg.model.roi_head.bbox_head[1].num_classes = 11
        cfg.model.roi_head.bbox_head[2].num_classes = 11
        return cfg
