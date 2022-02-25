from mmcv import Config
import os

main_path = '/media/palm/BiggerData/mmdetection/configs'

class Yolo:
    @staticmethod
    def generic():
        cfg = Config.fromfile(os.path.join(main_path, 'yolo/yolov3_d53_mstrain-608_273e_coco.py'))
        cfg.model.bbox_head.num_classes = 11
        return cfg


class RetinaNet:
    @staticmethod
    def resnet_50():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_r50_fpn_mstrain_640-800_3x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        return base_cfg

    @staticmethod
    def resnet_101():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_r101_fpn_mstrain_640-800_3x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
        return base_cfg

    @staticmethod
    def resnext_101_64():
        base_cfg = Config.fromfile(os.path.join(main_path, 'retinanet/retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py'))
        base_cfg.model.bbox_head.num_classes = 11
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
