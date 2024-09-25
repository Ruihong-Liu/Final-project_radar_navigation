import json
import torch
from tqdm import tqdm
import numpy as np
from data_loader_loader_main import RF3DPoseDataset,ToTensor,get_all_file_pairs
from Train_and_model_model import Simple3DConvModelWithTripleCNNFPNAndAttention,batch_rodrigues
from Train_and_model_plotting_3D_mesh import save_human_mesh,plot_comparison
from Train_and_model_main_train import ensure_directory_exists
import matplotlib.pyplot as plt
import trimesh
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import random

plt.switch_backend('Agg')

def compute_average_vertex_error(pred_vertices, gt_vertices):
    return np.mean(np.linalg.norm(pred_vertices - gt_vertices, axis=-1))

def compute_average_joint_localization_error(pred_joints, gt_joints):
    return np.mean(np.linalg.norm(pred_joints - gt_joints, axis=-1))

def compute_average_joint_rotation_error(pred_rotations, gt_rotations):
    return np.mean(np.linalg.norm(pred_rotations - gt_rotations, axis=(-1, -2)))

def compute_mesh_localization_error(pred_trans, gt_trans):
    return np.linalg.norm(pred_trans - gt_trans, axis=-1)

def compute_gender_prediction_accuracy(pred_genders, gt_genders):
    return (pred_genders > 0.5).eq(gt_genders.cuda()).float().mean().item()

def evaluate(model, dataloader, result_path):
    model.eval()
    frame_results = []
    vertex_errors = []
    joint_localization_errors = []
    joint_rotation_errors = []
    mesh_localization_errors = []
    gender_accuracies = []

    indices_to_save = random.sample(range(len(dataloader.dataset)), 10)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            vertices = batch['vertices'].cuda()
            gender = batch['gender'].cuda()

            rawImage_XYZ = rawImage_XYZ.unsqueeze(2)

            pred_betas, pred_pose, pred_root_orient, pred_trans, gender_pred = model(rawImage_XYZ)
            smplx_output = model.get_smplx_output(pred_betas, pred_pose, pred_root_orient, pred_trans, gender_pred)

            # Remove extra dimension if present
            smplx_output = smplx_output.squeeze(1)

            for i in range(rawImage_XYZ.size(0)):
                index = batch_idx * dataloader.batch_size + i
                vertex_error = compute_average_vertex_error(smplx_output[i].cpu().numpy(), vertices[i].cpu().numpy())
                vertex_errors.append(float(vertex_error))

                pred_joints = smplx_output[i].cpu().numpy()[:, :3]
                gt_joints = vertices[i].cpu().numpy()[:, :3]
                if pred_joints.shape == gt_joints.shape:
                    joint_localization_error = compute_average_joint_localization_error(pred_joints, gt_joints)
                    joint_localization_errors.append(float(joint_localization_error))
                else:
                    joint_localization_errors.append(float('nan'))

                pred_rotations = batch_rodrigues(torch.tensor(pred_pose[i].cpu().numpy().reshape(-1, 3))).reshape(-1, 3, 3)
                gt_rotations = batch_rodrigues(torch.tensor(vertices[i, :, 3:].cpu().numpy().reshape(-1, 3))).reshape(-1, 3, 3)
                if pred_rotations.shape == gt_rotations.shape:
                    joint_rotation_error = compute_average_joint_rotation_error(pred_rotations.cpu().numpy(), gt_rotations.cpu().numpy())
                    joint_rotation_errors.append(float(joint_rotation_error))
                else:
                    joint_rotation_errors.append(float('nan'))

                mesh_localization_error = compute_mesh_localization_error(pred_trans[i].cpu().numpy(), vertices[i, :, :3].cpu().numpy().mean(axis=0))
                mesh_localization_errors.append(float(mesh_localization_error))

                gender_accuracy = compute_gender_prediction_accuracy(gender_pred, gender)
                gender_accuracies.append(float(gender_accuracy))

                pred_vertices = smplx_output[i].cpu().numpy()
                gt_vertices = vertices[i].cpu().numpy()
                faces = model.faces_male if gender[i] > 0.5 else model.faces_female

                if np.max(faces) < pred_vertices.shape[0]:
                    if index in indices_to_save:
                        pred_filename = os.path.join(result_path, f'frame_{batch_idx}_{i}_pred.obj')
                        gt_filename = os.path.join(result_path, f'frame_{batch_idx}_{i}_gt.obj')
                        comparison_plot_filename = os.path.join(result_path, f'frame_{batch_idx}_{i}_comparison.png')

                        save_human_mesh(pred_vertices, faces, pred_filename)
                        save_human_mesh(gt_vertices, faces, gt_filename)
                        plot_comparison(pred_vertices, gt_vertices, model.faces_male, model.faces_female, gender.cpu().numpy(), i, comparison_plot_filename)
                else:
                    print(f"Skipping saving mesh for sample {i} due to invalid face indices.")

    overall_results = {
        'vertex_errors': vertex_errors,
        'joint_localization_errors': joint_localization_errors,
        'joint_rotation_errors': joint_rotation_errors,
        'mesh_localization_errors': mesh_localization_errors,
        'gender_accuracies': gender_accuracies
    }
    return frame_results, overall_results

def main():
    model_path = r'F:\code\Final-project_radar_navigation\results_Final9\best_model_fold_1.pth'
    smplx_model_paths = {
        'male': r'F:\code\Final-project_radar_navigation\models\SMPLX_MALE.npz',
        'female': r'F:\code\Final-project_radar_navigation\models\SMPLX_FEMALE.npz',
        'neutral': r'F:\code\Final-project_radar_navigation\models\SMPLX_NEUTRAL.npz'
    }
    model = Simple3DConvModelWithTripleCNNFPNAndAttention(smplx_model_paths, input_channels=31).cuda()
    model.load_state_dict(torch.load(model_path))

    root_dir = 'DataTesting'
    batch_size = 8

    file_pairs = get_all_file_pairs(root_dir)
    dataset = RF3DPoseDataset(file_pairs, transform=transforms.Compose([ToTensor()]))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    result_path = r'F:\code\Final-project_radar_navigation\results_Final9_evaluation'
    ensure_directory_exists(result_path)

    frame_results, overall_results = evaluate(model, test_loader, result_path)

    with open(os.path.join(result_path, 'evaluation_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)

if __name__ == "__main__":
    main()
