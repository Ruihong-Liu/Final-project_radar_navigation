import numpy as np
import scipy.io as sio
import json
import torch
import logging
from torch.utils.data import Dataset
import os
import re
from torchvision import transforms

log_file_path = 'data11_log.txt'

if log_file_path:
    directory = os.path.dirname(log_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path)
                    ])

logger = logging.getLogger('data_loader')
logger.setLevel(logging.INFO)

# Add gender info
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
        if np.isnan(rawImage_XYZ).any() or np.isinf(rawImage_XYZ).any():
            raise ValueError("NaN or Inf values found in rawImage_XYZ")
        return rawImage_XYZ
    except Exception as e:
        logging.error(f"Error loading MAT file {mat_path}: {e}")
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
        if np.isnan(vertices).any() or np.isinf(vertices).any():
            raise ValueError("NaN or Inf values found in vertices")
        return vertices, faces
    except Exception as e:
        logging.error(f"Error loading OBJ file {obj_path}: {e}")
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
        logging.error(f"Error loading JSON file {json_path}: {e}")
        return None

def get_all_file_pairs(root_dir):
    all_file_pairs = []
    sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'P\d+', d)]

    for sub_dir in sub_dirs:
        mat_root_dir = os.path.join(root_dir, sub_dir, 'actions')
        obj_root_dir = os.path.join(root_dir, sub_dir, f'{sub_dir.lower()}-objs')
        parameters_dir = os.path.join(root_dir, sub_dir, 'parameters')

        logging.info(f"Checking directory {sub_dir}:")
        if not os.path.exists(mat_root_dir) or not os.path.exists(obj_root_dir) or not os.path.exists(parameters_dir):
            logging.warning(f"Skipping directory {sub_dir}: Missing required paths.")
            continue

        for i in range(1, 51):
            json_path = os.path.join(parameters_dir, f'{i}.json')
            if not os.path.exists(json_path):
                logging.warning(f"Missing JSON file: {json_path}")
                continue

            json_data = load_json_data(json_path)
            mat_dir = os.path.join(mat_root_dir, str(i), 'mmwave')
            obj_dir = os.path.join(obj_root_dir, str(i))
            if not os.path.exists(mat_dir) or not os.path.exists(obj_dir):
                logging.warning(f"Skipping subdirectory {i} in {sub_dir}: Missing required paths.")
                continue

            mat_files = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.mat')]
            obj_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith('.obj')]
            matched_files = check_and_match_files(mat_files, obj_files)

            if not matched_files:
                continue

            gender = gender_info.get(sub_dir, 'unknown')

            for idx, (mat_file, obj_file) in enumerate(matched_files):
                all_file_pairs.append((mat_file, obj_file, json_path, idx, gender))

    return all_file_pairs

# Calibration transformation matrices
# gb_matrix = np.array([[375.66860062, 0.0, 319.99508973], 
#                       [0.0, 375.66347079, 239.41364796], 
#                       [0.0, 0.0, 1.0]])
# radar2rgb_tvec = np.array([-0.03981857, 1.35834002, -0.05225502])
# radar2rgb_rotmatrix = np.array([[9.99458797e-01, 3.28646073e-02, 1.42475954e-03], 
#                                 [4.78233954e-04, 2.87906567e-02, -9.99585349e-01], 
#                                 [-3.28919997e-02, 9.99045052e-01, 2.87593582e-02]])

# Calculate inverse rotation matrix and inverse translation vector
# rgb2radar_rotmatrix = np.linalg.inv(radar2rgb_rotmatrix)
# rgb2radar_tvec = -np.dot(radar2rgb_rotmatrix, radar2rgb_tvec)

# def transform_rgb_to_radar(points):
#     # Apply inverse rotation
#     transformed_points = np.dot(points, rgb2radar_rotmatrix.T)
#     # Apply inverse translation
#     transformed_points += rgb2radar_tvec
#     return transformed_points


class RF3DPoseDataset(Dataset):
    def __init__(self, file_pairs, transform=None):
        self.file_pairs = file_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        mat_file, obj_file, json_path, json_idx, gender = self.file_pairs[idx]

        rawImage_XYZ = load_mat_file(mat_file)
        vertices, faces = load_obj_file(obj_file)
        json_data = load_json_data(json_path)

        if rawImage_XYZ is None or vertices is None or json_data is None:
            logging.error(f"Error loading data at index {idx}: MAT, OBJ, or JSON file load issues.")
            return self.__getitem__((idx + 1) % len(self))

        joints = np.array(json_data['joints'][json_idx])
        betas = np.array(json_data['betas'][json_idx])
        pose_body = np.array(json_data['pose_body'][json_idx])
        trans = np.array(json_data['trans'][json_idx])
        root_orient = np.array(json_data['root_orient'][json_idx])

        if (np.isnan(joints).any() or np.isinf(joints).any() or
            np.isnan(betas).any() or np.isinf(betas).any() or
            np.isnan(pose_body).any() or np.isinf(pose_body).any() or
            np.isnan(trans).any() or np.isinf(trans).any() or
            np.isnan(root_orient).any() or np.isinf(root_orient).any()):
            logging.error(f"NaN or Inf found in sample data, skipping this sample index {idx}.")
            return self.__getitem__((idx + 1) % len(self))

        # Apply calibration transformation to ground truth (gt)
        # joints = transform_rgb_to_radar(joints)
        # vertices = transform_rgb_to_radar(vertices)
        # trans = transform_rgb_to_radar(trans)

        min_val = np.min(rawImage_XYZ)
        max_val = np.max(rawImage_XYZ)
        rawImage_XYZ = (rawImage_XYZ - min_val) / (max_val - min_val + 1e-8)

        rawImage_XYZ = rawImage_XYZ.transpose(2, 0, 1)

        sample = {
            'rawImage_XYZ': rawImage_XYZ,
            'vertices': vertices,
            'faces': faces,
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

        if not isinstance(rawImage_XYZ, torch.Tensor):
            rawImage_XYZ = torch.from_numpy(rawImage_XYZ).float()

        if not isinstance(gender, torch.Tensor):
            gender = torch.tensor(gender, dtype=torch.long)

        rawImage_XYZ = transforms.RandomHorizontalFlip()(rawImage_XYZ)
        rawImage_XYZ = transforms.RandomRotation(10)(rawImage_XYZ)

        return {
            'rawImage_XYZ': rawImage_XYZ,
            'vertices': torch.tensor(vertices).float(),
            'faces': torch.tensor(faces).long(),
            'joints': torch.tensor(joints).float(),
            'betas': torch.tensor(betas).float(),
            'pose_body': torch.tensor(pose_body).float(),
            'trans': torch.tensor(trans).float(),
            'root_orient': torch.tensor(root_orient).float(),
            'gender': gender.squeeze()
        }

class RF3DPoseDataset(Dataset):
    def __init__(self, file_pairs, transform=None):
        self.file_pairs = file_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        mat_file, obj_file, json_path, json_idx, gender = self.file_pairs[idx]

        rawImage_XYZ = load_mat_file(mat_file)
        vertices, faces = load_obj_file(obj_file)
        json_data = load_json_data(json_path)

        if rawImage_XYZ is None or vertices is None or json_data is None:
            logging.error(f"Error loading data at index {idx}: MAT, OBJ, or JSON file load issues.")
            return self.__getitem__((idx + 1) % len(self))

        joints = np.array(json_data['joints'][json_idx])
        betas = np.array(json_data['betas'][json_idx])
        pose_body = np.array(json_data['pose_body'][json_idx])
        trans = np.array(json_data['trans'][json_idx])
        root_orient = np.array(json_data['root_orient'][json_idx])

        if (np.isnan(joints).any() or np.isinf(joints).any() or
            np.isnan(betas).any() or np.isinf(betas).any() or
            np.isnan(pose_body).any() or np.isinf(pose_body).any() or
            np.isnan(trans).any() or np.isinf(trans).any() or
            np.isnan(root_orient).any() or np.isinf(root_orient).any()):
            logging.error(f"NaN or Inf found in sample data, skipping this sample index {idx}.")
            return self.__getitem__((idx + 1) % len(self))

        # Apply calibration transformation to ground truth (gt)
        # joints = transform_rgb_to_radar(joints)
        # vertices = transform_rgb_to_radar(vertices)
        # trans = transform_rgb_to_radar(trans)

        min_val = np.min(rawImage_XYZ)
        max_val = np.max(rawImage_XYZ)
        rawImage_XYZ = (rawImage_XYZ - min_val) / (max_val - min_val + 1e-8)

        rawImage_XYZ = rawImage_XYZ.transpose(2, 0, 1)

        sample = {
            'rawImage_XYZ': rawImage_XYZ,
            'vertices': vertices,
            'faces': faces,
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

def test_data_loader(dataset):
    total_pairs = len(dataset)
    logging.info(f"Total matched pairs: {total_pairs}")

    logging.info("First 10 matched file pairs:")
    for i in range(min(10, total_pairs)):
        sample = dataset.file_pairs[i]
        logging.info(f"Pair {i+1}:")
        logging.info(f"  MAT file: {sample[0]}")
        logging.info(f"  OBJ file: {sample[1]}")
        logging.info(f"  JSON file: {sample[2]}")
        logging.info(f"  Index: {sample[3]}")
        logging.info(f"  Gender: {sample[4]}")

    if total_pairs > 0:
        sample = dataset[0]
        logging.info("\nSample size details:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logging.info(f"  {key}: {value.size()}")
            else:
                logging.info(f"  {key}: {type(value)}")

if __name__ == "__main__":
    root_dir = "DataUsing1"
    file_pairs = get_all_file_pairs(root_dir)
    dataset = RF3DPoseDataset(file_pairs, transform=ToTensor())

    test_data_loader(dataset)
