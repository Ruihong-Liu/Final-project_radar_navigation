import os
import re
import numpy as np
import scipy.io as sio
import json
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
import torch
import logging
## 假设log_file_path已经定义在某处，并且不为空
log_file_path = 'data_log.txt'  # 确保log_file_path被正确设置

if log_file_path:  # 确保log_file_path不是空字符串
    directory = os.path.dirname(log_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# 配置日志记录器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path)
                    ])

# 添加用于数据加载的日志记录器
logger = logging.getLogger('data_loader')
logger.setLevel(logging.INFO)
# 确保只输出到文件
for handler in logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        logger.removeHandler(handler)
# Add gender info
gender_info = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1
}

# load mat file and find " rawImage_XYZ"
def load_mat_file(mat_path):
    try:
        mat_data = sio.loadmat(mat_path)
        if 'rawImage_XYZ' not in mat_data:
            raise KeyError(f"'{mat_path}' does not contain 'rawImage_XYZ'")
        rawImage_XYZ = mat_data['rawImage_XYZ']
        if np.isnan(rawImage_XYZ).any() or np.isinf(rawImage_XYZ).any():
            raise ValueError("NaN or Inf values found in rawImage_XYZ")
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
        if np.isnan(vertices).any() or np.isinf(vertices).any():
            raise ValueError("NaN or Inf values found in vertices")
        return vertices, faces
    except Exception as e:
        logger.error(f"Error loading OBJ file {obj_path}: {e}")
        return None, None


# sort the file for pairing
def sort_files_by_numeric_name(files, reverse=False):
    def numeric_key(filename):
        basename = os.path.basename(filename)
        number = re.findall(r'\d+', basename)
        return int(number[0]) if number else float('inf')
    return sorted(files, key=numeric_key, reverse=reverse)

# paring
def check_and_match_files(mat_files, obj_files):
    mat_files_sorted = sort_files_by_numeric_name(mat_files)
    obj_files_sorted = sort_files_by_numeric_name(obj_files)

    if len(mat_files_sorted) != len(obj_files_sorted):
        logging.warning(f"{len(mat_files_sorted)} .mat files and {len(obj_files_sorted)} .obj files found. Adjusting to match count.")
        min_len = min(len(mat_files_sorted), len(obj_files_sorted))
        mat_files_sorted = mat_files_sorted[-min_len:]
        obj_files_sorted = obj_files_sorted[-min_len:]

    matched_pairs = []
    for idx in range(len(mat_files_sorted)):
        mat_file = mat_files_sorted[idx]
        obj_file = obj_files_sorted[idx]
        matched_pairs.append((idx, mat_file, obj_file))

    return matched_pairs

# 加载JSON文件
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def match_files_with_json(matched_files, json_data, gender_info, sub_dir):
    matched_data = []
    gender = gender_info.get(sub_dir, None)
    if gender is None:
        logging.warning(f"Gender information not found for {sub_dir}.")
    
    for idx, mat_file, obj_file in matched_files:
        if idx < len(json_data['joints']):
            joints = np.array(json_data['joints'][idx])
            betas = np.array(json_data['betas'][idx])
            pose_body = np.array(json_data['pose_body'][idx])
            trans = np.array(json_data['trans'][idx])
            root_orient = np.array(json_data['root_orient'][idx])
            
            if (np.isnan(joints).any() or np.isinf(joints).any() or
                np.isnan(betas).any() or np.isinf(betas).any() or
                np.isnan(pose_body).any() or np.isinf(pose_body).any() or
                np.isnan(trans).any() or np.isinf(trans).any() or
                np.isnan(root_orient).any() or np.isinf(root_orient).any()):
                logging.error(f"Skipping index {idx} due to NaN or Inf values in JSON data. Details: "
                              f"joints: {np.isnan(joints).any()} {np.isinf(joints).any()}, "
                              f"betas: {np.isnan(betas).any()} {np.isinf(betas).any()}, "
                              f"pose_body: {np.isnan(pose_body).any()} {np.isinf(pose_body).any()}, "
                              f"trans: {np.isnan(trans).any()} {np.isinf(trans).any()}, "
                              f"root_orient: {np.isnan(root_orient).any()} {np.isinf(root_orient).any()}")
                continue
            
            matched_data.append({
                'index': idx,
                'mat_file': mat_file,
                'obj_file': obj_file,
                'joints': joints,
                'betas': betas,
                'pose_body': pose_body,
                'trans': trans,
                'root_orient': root_orient,
                'gender': gender
            })
        else:
            logging.error(f"Index {idx} out of range for JSON data.")
    
    return matched_data

# 分析数据集并进行配对
def analyze_dataset(root_dir):
    all_matched_data = []

    sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'P\d+', d)]
    
    for sub_dir in sub_dirs:
        folder_num = re.findall(r'\d+', sub_dir)[0]
        mat_root_dir = os.path.join(root_dir, sub_dir, 'actions')
        obj_root_dir = os.path.join(root_dir, sub_dir, f'p{folder_num}-objs')
        parameters_dir = os.path.join(root_dir, sub_dir, 'parameters')

        logging.info(f"Checking folder {sub_dir}:")
        logging.info(f"  mat_root_dir: {mat_root_dir}")
        logging.info(f"  obj_root_dir: {obj_root_dir}")
        logging.info(f"  parameters_dir: {parameters_dir}")

        if not os.path.exists(mat_root_dir):
            logging.warning(f"  Missing directory: {mat_root_dir}")
        if not os.path.exists(obj_root_dir):
            logging.warning(f"  Missing directory: {obj_root_dir}")
        if not os.path.exists(parameters_dir):
            logging.warning(f"  Missing directory: {parameters_dir}")

        if not os.path.exists(mat_root_dir) or not os.path.exists(obj_root_dir) or not os.path.exists(parameters_dir):
            logging.warning(f"Skipping folder {sub_dir}: one or more required paths are missing.")
            continue

        for i in range(1, 51):
            json_path = os.path.join(parameters_dir, f'{i}.json')
            if not os.path.exists(json_path):
                logging.warning(f"  Missing JSON file: {json_path}")
                continue

            json_data = load_json_data(json_path)

            mat_dir = os.path.join(mat_root_dir, str(i), 'mmwave')
            obj_dir = os.path.join(obj_root_dir, str(i))

            if not os.path.exists(mat_dir):
                logging.warning(f"  Missing mat_dir: {mat_dir}")
            if not os.path.exists(obj_dir):
                logging.warning(f"  Missing obj_dir: {obj_dir}")

            if not os.path.exists(mat_dir) or not os.path.exists(obj_dir):
                logging.warning(f"Skipping {sub_dir} sub-folder {i}: one or more required paths are missing.")
                continue

            mat_files = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.mat')]
            obj_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith('.obj')]

            matched_files = check_and_match_files(mat_files, obj_files)
            if not matched_files:
                continue

            matched_data = match_files_with_json(matched_files, json_data, gender_info, sub_dir)
            all_matched_data.extend(matched_data)

    return all_matched_data


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
        
        # Ensure rawImage_XYZ is converted properly
        if not isinstance(rawImage_XYZ, torch.Tensor):
            rawImage_XYZ = torch.from_numpy(rawImage_XYZ).float()

        return {
            'rawImage_XYZ': rawImage_XYZ,
            'vertices': torch.tensor(vertices).float(),
            'faces': torch.tensor(faces).long(),
            'joints': torch.tensor(joints).float(),
            'betas': torch.tensor(betas).float(),
            'pose_body': torch.tensor(pose_body).float(),
            'trans': torch.tensor(trans).float(),
            'root_orient': torch.tensor(root_orient).float(),
            'gender': torch.tensor(gender).long()
        }

class RF3DPoseDataset(Dataset):
    def __init__(self, matched_pairs, transform=None):
        self.matched_pairs = matched_pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        data = self.matched_pairs[idx]
        mat_path = data['mat_file']
        obj_path = data['obj_file']

        rawImage_XYZ = load_mat_file(mat_path)
        vertices, faces = load_obj_file(obj_path)
        
        # 如果MAT或OBJ文件加载出错，跳过这个索引并返回下一个有效索引的数据
        if rawImage_XYZ is None or vertices is None:
            logging.error(f"Error loading data for index {idx}: MAT or OBJ file loading issue.")
            return self.__getitem__((idx + 1) % len(self))

        # Normalize rawImage_XYZ to range [0, 1]
        min_val = np.min(rawImage_XYZ)
        max_val = np.max(rawImage_XYZ)
        rawImage_XYZ = (rawImage_XYZ - min_val) / (max_val - min_val + 1e-8)

        # Transpose to (C, D, H) format for 3D convolutions
        rawImage_XYZ = rawImage_XYZ.transpose(2, 0, 1)  # from (D, H, C) to (C, D, H)

        sample = {
            'rawImage_XYZ': rawImage_XYZ,  # Shape should be (C, D, H)
            'vertices': vertices,
            'faces': faces,
            'joints': data['joints'],
            'betas': data['betas'],
            'pose_body': data['pose_body'],
            'trans': data['trans'],
            'root_orient': data['root_orient'],
            'gender': data['gender']
        }

        # 再次检查 sample 中的 NaN 或 Inf 值
        for key, value in sample.items():
            if np.isnan(value).any() or np.isinf(value).any():
                logging.error(f"NaN or Inf found in sample data for index {idx}. Skipping this sample.")
                return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            sample = self.transform(sample)

        return sample
# Testing function
def test_data_loader(dataset):
    # 总的配对数量
    total_pairs = len(dataset)
    print(f"Total matched pairs: {total_pairs}")

    # 前10个配对文件的路径
    print("First 10 matched file pairs:")
    for i in range(min(10, total_pairs)):
        sample = dataset.matched_pairs[i]
        print(f"Pair {i+1}:")
        print(f"  MAT file: {sample['mat_file']}")
        print(f"  OBJ file: {sample['obj_file']}")

    # 一个样本的详细尺寸信息
    if total_pairs > 0:
        sample = dataset[0]
        print("\nSample size details:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.size()}")
            else:
                print(f"  {key}: {type(value)}")

# Main function to run the test
if __name__ == "__main__":
    root_dir = "DataUsing1"
    matched_data = analyze_dataset(root_dir)
    dataset = RF3DPoseDataset(matched_data, transform=ToTensor())

    test_data_loader(dataset)
