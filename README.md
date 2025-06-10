# FRFU: Flexible Receptive Field for Feature Upsampling

<p align="left"><img src="/reademeImage/感受野.png" width="500" title="Receptive field"/></p>
FRFU is a lightweight plug-and-play upsampling operator with task-agnostic applicability. FRFU establishes the feasibility of adaptive receptive fields for region-sensitive and detail-critical dense prediction tasks. FRFU achieves state-of-the-art performance while maintaining exceptional efficiency, imposing negligible computational overhead on real-time inference without requiring custom CUDA implementations. Crucially, as a single-feature upsampler, it attains high-frequency contour accuracy comparable to methods leveraging high-resolution guidance features, demonstrating that adaptive receptive fields hold significant promise in current upsampling applications.

:warning:Note: In the experimental logs and weight files of the project, the old name "rrfu" is used.


## Highlights

- **Simple code:** FRFU's code is concise and easy to modify;
- **Easy to use:** FRFU does not rely on any extra CUDA packages installed;
- **precise:** FRFU can efficiently sample both internal objects and contours.


## Upsampling results

<p align="left"><img src="/reademeImage/特征图.png" width="600" title="Example of Upsampling Feature Results"/></p>


## Work visualization

<p align="left"><img src="/reademeImage/工作原理图.png" width="800" title="FRFU working principle diagram"/></p>


## Experimental Result

**Table1: Semantic segmentation results with SegFormer on ADE20K**
| SegFormer-B1 | FLOPs    | Params   | mIoU     | bIoU       |  log  | ckpt  |
| ----------- | ------- | ------- | ------- | --------- | --- | --- |
| Nearest      | 15.9G    | 13.7M    | 40.54    | 24.64      |       |       |
| Bilinear     | 15.9G    | 13.7M    | 41.68    | 27.80      |       |       |
| FRFU         | +0.9G    | +0.1M    | **43.75** | **30.46**      |[Link](https://github.com/yisi-xj/frfu/blob/main/ADE20K_segmentation/segformer_mit-b1-rrfu_160k_ade20k-512x512_log.txt)||

**Table2: Object detection results with Faster R-CNN on MS COCO.**
| Faster R-CNN       | Backbone | Params   | AP   | AP₅₀ | AP₇₅ | APₛ  | APₘ  | APₗ  |  log  | ckpt  |
|--------------|----------|----------|------|------|------|------|------|------| --- | --- |
| Nearest      | R50      | 46.8M    | 37.5 | 58.2 | 40.8 | 21.3 | 41.1 | 48.9 |||
| **FRFU**     | R50      | +79.4K   | **39.1** | **60.5** | **42.5** | **22.8** | **43.0** | **50.5** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_detection/faster_rcnn_r50_fpn_rrfu_1x_coco.py_log.txt)|       |
| Nearest      | R101     | 65.8M    | 39.4 | 60.1 | 43.1 | 22.4 | 43.7 | 51.1 |       |       |
| **FRFU**     | R101     | +79.4K   | **40.7** | **61.8** | **44.6** | **24.4** | **44.8** | **53.1** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_detection/faster_rcnn_r101_fpn_rrfu_1x_coco.py_log.txt)|       |

**Table3: Instance segmentation results with Mask R-CNN on MS COCO.**
| Method       | Task | Backbone | AP   | AP₅₀ | AP₇₅ | APₛ  | APₘ  | APₗ  |  log  | ckpt  |
|--------------|------|----------|------|------|------|------|------|------| --- | --- |
| Nearest      | Bbox | R50      | 38.3 | 58.7 | 42.0 | 21.9 | 41.8 | 50.2 |       |       |
| **FRFU**     | Bbox | R50      | **40.0** | **60.9** | **43.8** | **23.8** | **43.6** | **52.0** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_instance/mask_rcnn_r50_fpn_rrfu_1x_coco_log.txt)|       |
| Nearest      | Bbox | R101     | 40.0 | 60.4 | 43.7 | 22.8 | 43.7 | 52.0 |       |       |
| **FRFU**     | Bbox | R101     | **41.2** | **62.1** | **45.2** | **24.5** | **45.3** | **54.1** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_instance/mask_rcnn_r101_fpn_rrfu_1x_coco_log.txt)|       |
| Nearest      | Segm | R50      | 34.7 | 55.8 | 37.2 | 16.7 | 37.3 | 50.8 |       |       |
| **FRFU**     | Segm | R50      | **36.1** | **57.6** | **38.4** | **17.5** | **38.7** | **52.2** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_instance/mask_rcnn_r50_fpn_rrfu_1x_coco_log.txt)|       |
| Nearest      | Segm | R101     | 36.0 | 57.6 | 38.5 | 16.5 | 39.3 | 52.2 |       |       |
| **FRFU**     | Segm | R101     | **37.0** | **58.9** | **39.6** | **17.9** | **40.3** | **54.3** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_instance/mask_rcnn_r101_fpn_rrfu_1x_coco_log.txt)|       |

**Table4: Panoptic segmentation results with Panoptic FPN on MS COCO.**
| Method       | Backbone | Params   | PQ   | PQᵗʰ | PQˢᵗ | SQ   | RQ   |  log  | ckpt  |
|--------------|----------|----------|------|------|------|------|------| --- | --- |
| Nearest      | R50      | 46.0M    | 40.2 | 47.8 | 28.9 | 77.8 | 49.3 |       |       |
| **FRFU**     | R50      | +63.1K   | **42.1** | **48.9** | **32.0** | **79.1** | **51.5** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_panoptic/panoptic-fpn_r50_fpn_rrfu_1x_coco_log.txt)|       |
| Nearest      | R101     | 65.0M    | 42.2 | 50.1 | 30.3 | 78.3 | 51.4 |       |       |
| **FRFU**     | R101     | +63.1K   | **43.3** | **50.4** | **32.8** | **79.5** | **52.7** |[Link](https://github.com/yisi-xj/frfu/blob/main/COCO2017_panoptic/panoptic-fpn_r101_fpn_rrfu_1x_coco_log.txt)|       |

**Table5: Monocular depth estimation results with DepthFormer on NYU Depth V2.**
| Method        | Params   | $\delta<1.25$ | Abs Rel | RMS   | log10 | RMS log | SI log | Sq Rel |  log  | ckpt  |
|---------------|----------|---------------|---------|-------|------------|--------------------|------------------|-------| --- | --- |
| Nearest       | 47.6M    | 0.856         | 0.128   | 0.445 | 0.053      | 0.159              | 13.00            | 0.085 |[Link](https://github.com/yisi-xj/frfu/blob/main/NYUv2_depth/depthformer_swint_w7_nyu_nearest_log.txt)|       |
| Bilinear      | 47.6M    | 0.856         | 0.127   | 0.445 | 0.053      | 0.159              | 13.00            | 0.084 |[Link](https://github.com/yisi-xj/frfu/blob/main/NYUv2_depth/depthformer_swint_w7_nyu_bilinear_log.txt)|       |
| **FRFU**      | +0.1M    | **0.865**     | **0.122** | **0.435** | **0.051** | **0.154**          | **12.62**        | **0.080** |[Link](https://github.com/yisi-xj/frfu/blob/main/NYUv2_depth/depthformer_swint_w7_nyu_rrfu-0.005_log.txt)|       |

## Usage

For application instances, one can refer to [detection-with-upsamplers](https://github.com/tiny-smart/detection-with-upsamplers), [segmentation-with-upsamplers](https://github.com/tiny-smart/segmentation-with-upsamplers) and [Monocular-Depth-Estimation-Toolbox
](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox) to try upsamplers with mmcv.

