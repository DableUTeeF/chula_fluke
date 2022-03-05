# A

|Technique|Backbone|Augment|Epoch|Finetune|LR Divider|Batch Size|Score|
|---|---|---|---|---|---|---|---|
|Cascade R-CNN|R50|Base|20|No|4|2|0.864|
|Cascade R-CNN|R50|Base|20|90/10|4/100|2|0.851|2|
|Cascade R-CNN|R101|Base|20|No|4|2|0.86|2|
|Cascade R-CNN|R101|Base|20|90/10|4/100*|2|0.88|2|
|Cascade R-CNN|X101-64|Base|12|No|4|2| |2|
|Cascade R-CNN|R101|Albu 1|20|90/10|4/100|2|0.86|2|
|Cascade R-CNN|PVTv2-B0|Base|20|No|1|2|0.889|2|
|Cascade R-CNN|PVTv2-B0|Base|20|90/10|1/25|2|0.891|2|
|VFNet|R50 DCN|Base|24|No|4|2|0.71|2|
|VFNet|R50|Base|20|90/10|4/100|2|0.71|2|
|Yolov3|D53|ms|20|No|4|6|0.823|6|
|Yolov3|D53|ms|20|90/10|4/100|6|0.819|6|
|Yolov3|D53|ms|20|90/10|4/10000|6|0.815|6|
|Yolov3|D53|ms, mixup|20|90/10|4/100|6|0.73908|6|
|Yolov3|D53|ms, mosaic|20|90/10|4/100|6|0.48041|6|
|Cascade RPN|R50|Base|20|90/10|4/100|2|0.864|2|
|RetinaNet|PVTv2-B0|Base|20|90/10|4/100|2|0.90702|2|
|RetinaNet|PVTv2-B1|Base|20|90/10|4/100|2|0.90979|2|
|RetinaNet|PVTv2-B3|Base|20|90/10|4/100|1|0.91109|1|
|RetinaNet|PVTv2-B3|Base|12|90/10|4/100|1|0.91109|1|
|RetinaNet|R101|Base|20|90/10|4/100|2|0.879|2|
|RetinaNet|RSB50|Base|20|90/10|4/100|2|0.14169**|2|
|GFL|R50|Base|20|90/10|4/100|2|0.879|2|
|Tood|R50|Base|20|90/10|4/100|2|0.8563|2|

\* No model re-initiate

** The cls_loss was really high
