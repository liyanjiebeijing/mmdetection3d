from mmdet3d.apis import init_model, inference_detector, show_result_meshlab
import open3d as o3d
from mmdet3d.core import Box3DMode
import numpy as np

# 1. 初始化模型
config_file = 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'
checkpoint_file = 'work_dirs/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/epoch_24.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 2. 推理
pointcloud_file = 'data/nuscenes/samples/LIDAR_TOP/n015-2018-08-01-16-32-59+0800__LIDAR_TOP__1533112831147007.pcd.bin'
result, data = inference_detector(model, pointcloud_file)

# 3. 可视化

#点云着色
points_tensor = data['points'][0][0]  # torch.Tensor
points = points_tensor[:, :3].cpu().numpy()  # (N,3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1.0, 1.0, 1.0])   # 灰色

#过滤分数较低的框
if 'pts_bbox' in result[0].keys():
    pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
else:
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

# filter out low score bboxes for visualization
score_thr = 0.5
if score_thr > 0:
    pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
    inds = pred_scores > score_thr
    pred_bboxes = pred_bboxes[inds]


def create_dense_lineset_from_bbox(center, size, yaw, density=0.1, color=[1,0,0]):
    """
    根据3D框参数生成稠密点云 (只包含12根线)
    center: (x,y,z)
    size: (dx,dy,dz)
    yaw: 绕z轴旋转角度（弧度）
    density: 点间隔，越小越稠密
    color: RGB 0-1
    """
    x, y, z = center
    dx, dy, dz = size

    # 8个角点（局部坐标）
    corners = np.array([
        [ dx/2,  dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
        [ dx/2,  dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
    ])

    # 旋转+平移到全局
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]])
    corners = corners @ rot.T + np.array([x,y,z])

    # 12条边 (Open3D线框的顺序)
    lines = [
        [0,1],[1,2],[2,3],[3,0],  # 上面
        [4,5],[5,6],[6,7],[7,4],  # 下面
        [0,4],[1,5],[2,6],[3,7]   # 竖线
    ]

    # 在每条线段上均匀采样
    all_points = []
    for start_idx, end_idx in lines:
        p1 = corners[start_idx]
        p2 = corners[end_idx]
        length = np.linalg.norm(p2 - p1)
        n_points = max(int(length / density), 2)
        t = np.linspace(0, 1, n_points)
        line_points = (1-t)[:,None]*p1 + t[:,None]*p2
        all_points.append(line_points)

    all_points = np.vstack(all_points)

    # 转为Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.paint_uniform_color(color)

    return pcd


# 为每个3D框生成稠密点云
bbox_pcds = []
for bbox in pred_bboxes:
    center = bbox[:3]
    size = bbox[3:6]
    yaw = bbox[6]
    bbox_pcds.append(create_dense_lineset_from_bbox(center, size, yaw, density=0.05, color=[1,0,0]))

# 合并点云和3D框点云
merged = pcd
for b in bbox_pcds:
    merged += b

# 保存为PCD文件
o3d.io.write_point_cloud("vis_results/detection_bbox.pcd", merged)

print("保存完成：pointcloud_and_boxes.pcd")