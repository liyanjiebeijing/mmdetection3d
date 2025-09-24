from mmdet3d.datasets import Custom3DDataset
import numpy as np
from .builder import DATASETS

@DATASETS.register_module()
class FakeDataset3D(Custom3DDataset):
    CLASSES = ('Car',)

    def __init__(self, num_samples=10, with_gt=True, **kwargs):
        self.num_samples = num_samples
        self.with_gt = with_gt
        super().__init__(ann_file=None, pipeline=[], **kwargs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 构造一个假的点云
        points = np.random.rand(1000, 4).astype(np.float32)  # [x,y,z,intensity]
        # 构造一个假的bbox
        if self.with_gt:
            gt_bboxes_3d = np.random.rand(1, 7).astype(np.float32)  # x,y,z,dx,dy,dz,yaw
            gt_labels_3d = np.array([0], dtype=np.int64)
        else:
            gt_bboxes_3d = None
            gt_labels_3d = None
        return dict(points=[points], gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
