# A

|Technique|Backbone|Augment|Epoch|Finetune|LR Divider|Score|
|---|---|---|---|---|---|---|
|Cascade R-CNN|R50|Base|20|No|4| |
|Cascade R-CNN|R50|Base|20|90/10|4/100|0.851|
|Cascade R-CNN|R101|Base|20|No|4|0.86|
|Cascade R-CNN|R101|Base|20|90/10|4/100*|0.88|
|Cascade R-CNN|R101|Albu 1|20|90/10|4/100|0.86|
|VFNet|R50 DCN|Base|24|No|4|0.71|
|Cascade R-CNN|PVTv2-B0|Base|20|No|1|0.889|
|Cascade R-CNN|PVTv2-B0|Base|20|90/10|1/25|0.891|
|Yolov3|D53|Base|20|No|4|0.823|
|Yolov3|D53|Base|20|90/10|4/100|0.819|
|Cascade RPN|R50|Base|12|90/10|4/100|0.864|
|RetinaNet|PVTv2-B0|Base|12|90/10|4/100| |

\* No model re-initiate
