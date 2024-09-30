import os.path
import xml
from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
import numpy as np

# details of camera normalization can be found in Sec. C.3 in https://openaccess.thecvf.com/content/CVPR2023/supplemental/Cao_Multi-View_Azimuth_Stereo_CVPR_2023_supplemental.pdf
def normalize_camera(R_list, t_list, camera2object_ratio=3):
    A_camera_normalize = 0
    b_camera_normalize = 0
    camera_center_list = []
    for view_idx in range(len(R_list)):
        R = R_list[view_idx]
        t = t_list[view_idx]
        camera_center = - R.T @ t  # in world coordinate
        camera_center_list.append(camera_center)
        vi = R[2][:, None]  # the camera's principal axis in the world coordinates
        Vi = vi @ vi.T
        A_camera_normalize += np.eye(3) - Vi
        b_camera_normalize += camera_center.T @ (np.eye(3) - Vi)
    offset = np.linalg.lstsq(A_camera_normalize, np.squeeze(b_camera_normalize), rcond=None)[0]
    camera_center_dist_list = [np.sqrt(np.sum((np.squeeze(c) - offset) ** 2))
                               for c in camera_center_list]
    scale = np.max(camera_center_dist_list) / camera2object_ratio
    return offset, scale

def make4x4(P):
    assert P.shape[-1] == 4 or P.shape[-1] == 3
    assert len(P.shape) == 2
    assert P.shape[0] == 3 or P.shape[0] == 4
    ret = np.eye(4)
    ret[:P.shape[0], :P.shape[1]] = P
    return ret

class MetashapePoseLoader:
    def __init__(self, xml_path, camera2object_ratio):
        with open(xml_path, "r") as f:
            xml_data = f.read()
        bs_data = BeautifulSoup(xml_data, "xml")
        c_unique = bs_data.find_all('resolution')
        img_width = int(c_unique[0].get("width"))
        img_height = int(c_unique[0].get("height"))
        c_intrinsics = bs_data.find_all('calibration')
        f = float(c_intrinsics[0].find("f").text)
        cx_offset = float(c_intrinsics[0].find("cx").text)
        cy_offset = float(c_intrinsics[0].find("cy").text)
        K = np.array([[f, 0, (img_width-1)/2 + cx_offset],
                        [0, f, (img_height-1)/2 + cy_offset],
                        [0, 0, 1]])

        b_unique = bs_data.find_all('camera')
        R_list = []
        t_list = []
        C2W_list = []
        camera_sphere = dict()
        for tag in b_unique:
            img_name = tag.get("label")
            view_idx = int(img_name.split("_")[-1])
            # camera to world transform
            C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))
            C2W_list.append(C2W)

            assert int(img_name) == view_idx

            W2C = np.linalg.inv(C2W)
            R_list.append(W2C[:3, :3])
            t_list.append(W2C[:3, 3])

            camera_sphere[f"world_mat_{view_idx}"] = make4x4(K) @ W2C

        offset, scale = normalize_camera(R_list, t_list, camera2object_ratio=camera2object_ratio)
        print("offset", offset, "scale", scale)
        num_views = len(C2W_list)

        scale_mat = np.eye(4)
        scale_mat[:3, :3] *= scale
        scale_mat[:3, 3] = offset
        for im_idx in range(num_views):
            camera_sphere[f"scale_mat_{im_idx}"] = scale_mat

        data_dir = os.path.dirname(xml_path)
        np.savez(os.path.join(data_dir, 'cameras_sphere.npz'), **camera_sphere)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=10)
    args = parser.parse_args()

    MetashapePoseLoader(args.xml_path, camera2object_ratio=args.ratio)