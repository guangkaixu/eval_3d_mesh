'''
The evaluation code is modified from two github repos:
1. https://github.com/autonomousvision/convolutional_occupancy_networks/blob/f15b97da2c5995598537c8f832e52e95c0b09236/src/eval.py
2. https://github.com/nihalsid/retrieval-fuse/blob/main/util/mesh_metrics.py
'''

import logging
import numpy as np
import trimesh
from scipy.spatial import cKDTree
# from src.common import compute_iou

# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
    ''' Mesh evaluation class.
    It handles the mesh evaluation process.
    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, mesh_tgt, pointcloud, pointcloud_tgt, normals, normals_tgt):
        ''' Evaluates a mesh.
        Args:
            mesh (trimesh): mesh which should be evaluated
            mesh_tgt (trimesh): target mesh
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        out_dict['iou'] = compute_iou(mesh, mesh_tgt)

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1./1000, 1, 1000)):
        ''' Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score': F[9], # threshold = 1.0%
            'f-score-15': F[14], # threshold = 1.5%
            'f-score-20': F[19], # threshold = 2.0%
        }

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def compute_iou(mesh_pred, mesh_target):
    res = 1.1875
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)

    v_pred_filled = set(tuple(x) for x in v_pred.points)
    v_target_filled = set(tuple(x) for x in v_target.points)
    iou = len(v_pred_filled.intersection(v_target_filled)) / len(v_pred_filled.union(v_target_filled))
    return iou


def evaluate_mesh(mesh_pred_path, mesh_tgt_path):

    # load mesh
    mesh_pred = trimesh.load(mesh_pred_path)
    mesh_tgt = trimesh.load(mesh_tgt_path)
    normals_pred = mesh_pred.face_normals
    normals_tgt = mesh_tgt.face_normals

    # sample point cloud from mesh
    sample_num = max(len(normals_pred), len(normals_tgt)) # try to sample all of points
    pointcloud_pred, inx_pred = mesh_pred.sample(sample_num, return_index=True)
    pointcloud_tgt, inx_tgt = mesh_tgt.sample(sample_num, return_index=True)
    normals_pred = normals_pred[inx_pred]
    normals_tgt = normals_tgt[inx_tgt]

    # evaluate
    evaluator = MeshEvaluator()
    eval_dict_mesh = evaluator.eval_mesh(mesh_pred, mesh_tgt, pointcloud_pred, pointcloud_tgt, normals_pred, normals_tgt)

    return eval_dict_mesh


if __name__ == '__main__':

    mesh_pred_path = './gt_mesh.ply'
    mesh_tgt_path = './gt_mesh.ply'

    eval_dict_mesh = evaluate_mesh(mesh_pred_path, mesh_tgt_path)
    print('eval_dict_mesh :', eval_dict_mesh)
