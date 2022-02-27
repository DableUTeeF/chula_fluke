from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import models
import numpy as np
import os
import json
import time


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
if __name__ == '__main__':
    data = '/media/palm/BiggerData/Chula_Parasite/test/data'
    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py')
    cfg.model.roi_head.bbox_head[0].num_classes = 11
    cfg.model.roi_head.bbox_head[1].num_classes = 11
    cfg.model.roi_head.bbox_head[2].num_classes = 11

    # cfg.model.backbone = dict(
    #     # _delete_=True,
    #     type='PyramidVisionTransformerV2',
    #     embed_dims=32,
    #     num_layers=[2, 2, 2, 2],
    #     init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
    #                              'releases/download/v2/pvt_v2_b0.pth'))
    #
    # cfg.model.neck.in_channels = [32, 64, 160, 256]

    model = init_detector(models.CascadeRCNN.crpn(), '/media/palm/BiggerData/Chula_Parasite/checkpoints/crpn_r50_ft/epoch_20.pth', device='cuda')
    outputs = {}
    annotations = []
    for i, file in enumerate(os.listdir(data)):
        if not file.endswith('.jpg'):
            print(file)
            continue
        print(i)
        filename = os.path.join(data, file)
        # img = cv2.imread(filename)
        results = inference_detector(model, filename)
        # show_result_pyplot(model, filename, results)
        for idx, class_name in enumerate(CLASSES):
            bboxes = results[idx].astype(float)
            if bboxes.shape[0] == 0:
                continue
            for bbox in bboxes:
                # print('a')
                if bbox[-1] < 0.5:
                    continue
                x1, y1, x2, y2, _ = bbox.tolist()
                output = {
                    'id': len(annotations),
                    'file_name': file,
                    'category_id': idx,
                    'bbox': [x1, y1, x2-x1, y2-y1]
                }
                annotations.append(output)
    outputs['annotations'] = annotations
    f = str(time.time())
    os.makedirs(os.path.join('results', f))
    json.dump(outputs, open(f'results/{f}/result_crpn_r50_ft.json', 'w'))
