import logging
import os
import numpy as np
# set log information path
log_file_path = 'data11_log_cali.txt'

if log_file_path:
    directory = os.path.dirname(log_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path)])
logger = logging.getLogger('data_loader')
logger.setLevel(logging.INFO)

# comvert vicon frame to camera frame
def vicon_to_cam_transform(vicon_coords, vicon_to_cam_rotmatrix, vicon_to_cam_tvec):
    try:
        cam_coords = np.dot(vicon_to_cam_rotmatrix, vicon_coords.T).T + vicon_to_cam_tvec
        return cam_coords
    except Exception as e:
        logger.error(f"Error during Vicon to Camera transformation: {e}")
        return None

#convert camera frame to radar frame
def cam_to_radar_transform(cam_coords, radar_to_cam_rotmatrix, radar_to_cam_tvec):
    try:
        # 逆变换
        radar_coords = np.dot(np.linalg.inv(radar_to_cam_rotmatrix), (cam_coords - radar_to_cam_tvec).T).T
        return radar_coords
    except Exception as e:
        logger.error(f"Error during Camera to Radar transformation: {e}")
        return None