import numpy as np
import scipy.io as sio
import json
import torch
import os
import re
import matplotlib.pyplot as plt
import random
import logging
from torchvision import transforms
from torch.utils.data import Dataset

# 设置日志文件路径
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

# 添加性别信息
gender_info = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1
}

def load_mat_file(mat_path):
    try:
        mat_data = sio.loadmat(mat_path)
        rawImage_XYZ = mat_data.get('rawImage_XYZ')
        if rawImage_XYZ is None:
            raise ValueError(f"'rawImage_XYZ' not found in {mat_path}")
        rawImage_XYZ = rawImage_XYZ.astype(np.float32)
        return rawImage_XYZ
    except Exception as e:
        logger.error(f"Error loading MAT file {mat_path}: {e}")
        return None

def load_obj_file(obj_path):
    try:
        vertices = []
        faces = []
        with open(obj_path, 'r') as obj_file:
            for line in obj_file:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append([int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]])
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
        return vertices, faces
    except Exception as e:
        logger.error(f"Error loading OBJ file {obj_path}: {e}")
        return None, None

def extract_number(filename):
    match = re.findall(r'\d+', filename)
    return int(match[0]) if match else float('inf')

def check_and_match_files(mat_files, obj_files):
    mat_files.sort(key=lambda x: extract_number(os.path.basename(x)))
    obj_files.sort(key=lambda x: extract_number(os.path.basename(x)))
    
    matched_files = list(zip(mat_files, obj_files))
    return matched_files

def load_json_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {json_path}: {e}")
        return None

def load_transformation_matrices(ins_ext_dir):
    try:
        cam_ins = np.loadtxt(os.path.join(ins_ext_dir, 'cam_ins.txt'))
        vicon_to_cam = np.loadtxt(os.path.join(ins_ext_dir, 'vicon_to_cam.txt'))
        
        radar_to_cam_rotmatrix = np.loadtxt(os.path.join(ins_ext_dir, 'radar2rgb_rotmatrix.txt'))
        radar_to_cam_tvec = np.loadtxt(os.path.join(ins_ext_dir, 'radar2rgb_tvec.txt'))
        
        vicon_to_cam_rotmatrix = vicon_to_cam[:3, :3]
        vicon_to_cam_tvec = vicon_to_cam[:3, 3]
        
        return cam_ins, vicon_to_cam_rotmatrix, vicon_to_cam_tvec, radar_to_cam_rotmatrix, radar_to_cam_tvec
    except Exception as e:
        logger.error(f"Error loading transformation matrices from {ins_ext_dir}: {e}")
        return None, None, None, None, None


def vicon_to_cam_transform(vicon_coords, vicon_to_cam_rotmatrix, vicon_to_cam_tvec):
    try:
        cam_coords = np.dot(vicon_to_cam_rotmatrix, vicon_coords.T).T + vicon_to_cam_tvec
        return cam_coords
    except Exception as e:
        logger.error(f"Error during Vicon to Camera transformation: {e}")
        return None

def cam_to_radar_transform(cam_coords, radar_to_cam_rotmatrix, radar_to_cam_tvec):
    try:
        # 逆变换
        radar_coords = np.dot(np.linalg.inv(radar_to_cam_rotmatrix), (cam_coords - radar_to_cam_tvec).T).T
        return radar_coords
    except Exception as e:
        logger.error(f"Error during Camera to Radar transformation: {e}")
        return None

def project_to_2d(points_3d, camera_matrix, camera_extrinsics):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_camera = np.dot(camera_extrinsics, points_3d_homogeneous.T).T
    points_image_homogeneous = np.dot(camera_matrix, points_camera[:, :3].T).T
    points_2d = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2, np.newaxis]
    return points_2d

class ToTensor(object):
    def __call__(self, sample):
        rawImage_XYZ = sample['rawImage_XYZ']
        vertices = sample['vertices']
        faces = sample['faces']
        joints = sample['joints']
        betas = sample['betas']
        pose_body = sample['pose_body']
        trans = sample['trans']
        root_orient = sample['root_orient']
        gender = sample['gender']
        projected_vertices = sample['projected_vertices']
        image = sample['image']

        # 保证转换为torch Tensor
        rawImage_XYZ = torch.tensor(rawImage_XYZ).float()
        vertices = torch.tensor(vertices).float()  # 这里的vertices已经是新的经过转换的顶点数据
        faces = torch.tensor(faces).long()
        joints = torch.tensor(joints).float()
        betas = torch.tensor(betas).float()
        pose_body = torch.tensor(pose_body).float()
        trans = torch.tensor(trans).float()
        root_orient = torch.tensor(root_orient).float()
        gender = torch.tensor(gender, dtype=torch.long)

        if image is not None:
            image = torch.tensor(image).float().permute(2, 0, 1)  # HWC to CHW

        return {
            'rawImage_XYZ': rawImage_XYZ,
            'vertices': vertices,  # 使用新的Tensor化的vertices
            'faces': faces,
            'joints': joints,
            'betas': betas,
            'pose_body': pose_body,
            'trans': trans,
            'root_orient': root_orient,
            'gender': gender,
            'projected_vertices': torch.tensor(projected_vertices).float(),
            'image': image
        }

def get_all_file_pairs(root_dir):
    all_file_pairs = []
    sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'P\d+', d)]

    for sub_dir in sub_dirs:
        mat_root_dir = os.path.join(root_dir, sub_dir, 'actions')
        obj_root_dir = os.path.join(root_dir, sub_dir, f'{sub_dir.lower()}-objs')
        parameters_dir = os.path.join(root_dir, sub_dir, 'parameters')

        logger.info(f"Checking directory {sub_dir}:")
        logger.info(f"  Expected mat_root_dir: {mat_root_dir}")
        logger.info(f"  Expected obj_root_dir: {obj_root_dir}")
        logger.info(f"  Expected parameters_dir: {parameters_dir}")

        if not (os.path.exists(mat_root_dir) and os.path.exists(obj_root_dir) and os.path.exists(parameters_dir)):
            logger.warning(f"Skipping directory {sub_dir}: Missing required paths.")
            continue

        for i in range(1, 51):
            json_path = os.path.join(parameters_dir, f'{i}.json')
            if not os.path.exists(json_path):
                logger.warning(f"Missing JSON file: {json_path}")
                continue

            mat_dir = os.path.join(mat_root_dir, str(i), 'mmwave')
            rgb_dir = os.path.join(mat_root_dir, str(i), 'rgb')
            obj_dir = os.path.join(obj_root_dir, str(i))

            logger.info(f"  Expected mat_dir: {mat_dir}")
            logger.info(f"  Expected obj_dir: {obj_dir}")
            logger.info(f"  Expected rgb_dir: {rgb_dir}")

            if not (os.path.exists(mat_dir) and os.path.exists(obj_dir) and os.path.exists(rgb_dir)):
                logger.warning(f"Skipping subdirectory {i} in {sub_dir}: Missing required paths.")
                continue

            mat_files = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.mat')]
            obj_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith('.obj')]
            png_files = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')]

            matched_files = check_and_match_files(mat_files, obj_files)

            if not matched_files:
                logger.warning(f"No matching .mat and .obj files found in {mat_dir} and {obj_dir}")
                continue

            gender = gender_info.get(sub_dir, 'unknown')

            for idx, (mat_file, obj_file) in enumerate(matched_files):
                png_file = png_files[idx] if idx < len(png_files) else None
                all_file_pairs.append((mat_file, obj_file, png_file, json_path, idx, gender))

    return all_file_pairs

class RF3DPoseDataset(Dataset):
    def __init__(self, file_pairs, transform=None):
        self.file_pairs = file_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = len(self.file_pairs)
        
        while attempts < max_attempts:
            try:
                mat_file, obj_file, png_file, json_path, json_idx, gender = self.file_pairs[idx]
                rawImage_XYZ = load_mat_file(mat_file)
                vertices, faces = load_obj_file(obj_file)
                json_data = load_json_data(json_path)

                # 检查是否有任何数据为 None，如果有，则跳过此数据对
                if rawImage_XYZ is None or vertices is None or faces is None or json_data is None:
                    logger.error(f"Error loading data at index {idx}: MAT, OBJ, or JSON file load issues.")
                    idx = (idx + 1) % len(self.file_pairs)
                    attempts += 1
                    continue  # 尝试下一个索引

                vertices *= 1000

                ins_ext_dir = os.path.join(os.path.dirname(json_path), '../ins_ext')
                cam_ins, vicon_to_cam_rotmatrix, vicon_to_cam_tvec, radar_to_cam_rotmatrix, radar_to_cam_tvec = load_transformation_matrices(ins_ext_dir)

                if cam_ins is None or vicon_to_cam_rotmatrix is None or vicon_to_cam_tvec is None or radar_to_cam_rotmatrix is None or radar_to_cam_tvec is None:
                    logger.error(f"Error loading transformation matrices for index {idx}")
                    idx = (idx + 1) % len(self.file_pairs)
                    attempts += 1
                    continue  # 尝试下一个索引

                cam_coords = vicon_to_cam_transform(vertices, vicon_to_cam_rotmatrix, vicon_to_cam_tvec)
                if cam_coords is None:
                    logger.error(f"Error transforming vertices from Vicon to Camera coordinates for index {idx}")
                    idx = (idx + 1) % len(self.file_pairs)
                    attempts += 1
                    continue  # 尝试下一个索引

                radar_coords = cam_to_radar_transform(cam_coords, radar_to_cam_rotmatrix, radar_to_cam_tvec)
                if radar_coords is None:
                    logger.error(f"Error transforming vertices from Camera to Radar coordinates for index {idx}")
                    idx = (idx + 1) % len(self.file_pairs)
                    attempts += 1
                    continue  # 尝试下一个索引

                camera_extrinsics = np.eye(4)
                camera_extrinsics[:3, :3] = radar_to_cam_rotmatrix
                camera_extrinsics[:3, 3] = radar_to_cam_tvec

                projected_vertices = project_to_2d(radar_coords, cam_ins[:3, :3], camera_extrinsics)
                if projected_vertices is None:
                    logger.error(f"Error projecting vertices to 2D for index {idx}")
                    idx = (idx + 1) % len(self.file_pairs)
                    attempts += 1
                    continue  # 尝试下一个索引

                image = plt.imread(png_file) if png_file else None

                try:
                    joints = np.array(json_data['joints'][json_idx])
                    betas = np.array(json_data['betas'][json_idx])
                    pose_body = np.array(json_data['pose_body'][json_idx])
                    trans = np.array(json_data['trans'][json_idx])
                    root_orient = np.array(json_data['root_orient'][json_idx])
                except KeyError as e:
                    logger.error(f"Missing key {e} in JSON data for index {idx}")
                    idx = (idx + 1) % len(self.file_pairs)
                    attempts += 1
                    continue  # 尝试下一个索引

                sample = {
                    'rawImage_XYZ': rawImage_XYZ,
                    'vertices': radar_coords,
                    'faces': faces,
                    'projected_vertices': projected_vertices,
                    'image': image,
                    'joints': joints,
                    'betas': betas,
                    'pose_body': pose_body,
                    'trans': trans,
                    'root_orient': root_orient,
                    'gender': gender
                }

                if self.transform:
                    sample = self.transform(sample)

                return sample
            except Exception as e:
                logger.error(f"Error in __getitem__ at index {idx}: {e}")
                idx = (idx + 1) % len(self.file_pairs)
                attempts += 1
                continue
        
        # 如果所有尝试都失败，返回 None 或引发异常
        logger.error(f"All attempts failed for index {idx}.")
        raise RuntimeError(f"Unable to load a valid data pair after {max_attempts} attempts.")

def plot_projected_image(image, projected_vertices, sample_idx):
    try:
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            image = image.transpose(1, 2, 0)  # CHW to HWC

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
        plt.title(f'Projected Vertices on Image {sample_idx}')
        plt.axis('off')
        plt.savefig(f"projected_image_{sample_idx}.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error during plotting projected image for sample {sample_idx}: {e}")

def plot_random_samples(dataset, num_samples=5):
    try:
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            sample = dataset[idx]
            if sample is not None:
                plot_projected_image(sample['image'], sample['projected_vertices'], idx)
    except Exception as e:
        logger.error(f"Error during random sample plotting: {e}")

def test_data_loader(dataset):
    try:
        plot_random_samples(dataset, num_samples=5)
    except Exception as e:
        logger.error(f"Error during test data loader execution: {e}")

if __name__ == "__main__":
    try:
        root_dir = "/mnt/data-B/Ruihong_radar/DataUsing1"
        file_pairs = get_all_file_pairs(root_dir)
        dataset = RF3DPoseDataset(file_pairs, transform=ToTensor())
        test_data_loader(dataset)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
