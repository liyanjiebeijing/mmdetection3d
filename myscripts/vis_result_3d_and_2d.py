from mmdet3d.apis import init_model, inference_detector
from mmdet3d.apis import show_result_meshlab

config_file = 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'
checkpoint_file = 'work_dirs/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/epoch_24.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

pcd_path = 'data/nuscenes/samples/LIDAR_TOP/n015-2018-08-01-16-32-59+0800__LIDAR_TOP__1533112831147007.pcd.bin'
result, data = inference_detector(model, pcd_path)
# print(result)
# print('-' * 50)
# print(data)

out_dir = 'vis_results'
show_result_meshlab(
    data,                  # 上面 inference_detector 的 data
    result,                # 上面 inference_detector 的 result
    out_dir=out_dir,       # 输出目录
    score_thr=0.3)         # 分数阈值



from mmdet3d.core import show_multi_modality_result

import pdb
pdb.set_trace()
img_metas = data['img_metas'][0][0]
points = data['points'][0][0]

# 遍历6个相机视角
for cam_idx, img in enumerate(data['img'][0][0]):
    show_multi_modality_result(
        img=img,                            # 相机图像
        pts=points,                         # 点云
        pred_bboxes_3d=result[0]['boxes_3d'],
        pred_labels=result[0]['labels_3d'],
        img_metas=img_metas,                # 相机内外参
        out_dir=f'vis_results/cam_{cam_idx}',  # 保存目录
        show=False)                        # 不弹窗口，直接保存