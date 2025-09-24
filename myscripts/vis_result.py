from mmdet3d.apis import init_model, inference_detector, show_result_meshlab

# 1. 初始化模型
config_file = 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'
checkpoint_file = 'work_dirs/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/epoch_24.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 2. 推理
pointcloud_file = 'data/nuscenes/samples/LIDAR_TOP/n015-2018-08-01-16-32-59+0800__LIDAR_TOP__1533112831147007.pcd.bin'
result, data = inference_detector(model, pointcloud_file)

# 3. 可视化
show_result_meshlab(
    data,             # 输入数据（点云）
    result,           # 检测结果
    out_dir='vis_results',  # 输出目录
    score_thr=0.5,
    show=False         # 如果有GUI环境可以弹窗
)
