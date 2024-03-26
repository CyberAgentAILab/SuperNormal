from scipy.spatial import KDTree
import numpy as np


def chamfer_distance_and_f1_score(ref_points, eval_points, f_threshold=0.5):
    """
    This function calculates the chamfer distance and f1 score between two sets of points.

    Parameters:
    ref_points (numpy.ndarray): Reference points. A (p, 3) array representing points in the world space.
    eval_points (numpy.ndarray): Points to be evaluated. A (p, 3) array representing points in the world space.
    f_threshold (float, optional): Threshold for f1 score calculation. Default is 0.5mm.

    Returns:
    chamfer_dist (float): The chamfer distance between gt_points and eval_points.
    f_score (float): The f1 score between gt_points and eval_points.
    """
    print("computing chamfer distance and f1 score...")
    distance_eval2gt, _ = KDTree(ref_points).query(eval_points, k=1, p=2)   # p=2 for Euclidean distance
    distance_gt2eval, _ = KDTree(eval_points).query(ref_points, k=1, p=2)

    # following Uncertainty-aware deep multi-view photometric stereo
    chamfer_dist = (np.mean(distance_eval2gt) + np.mean(distance_gt2eval))/2

    precision = np.mean(distance_eval2gt < f_threshold)
    recall = np.mean(distance_gt2eval < f_threshold)
    f_score = 2 * precision * recall / (precision + recall)

    return chamfer_dist, f_score
