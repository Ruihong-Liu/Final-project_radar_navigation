import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import RF3DPoseDataset
from Main85 import Simple3DConvModel, save_obj, batch_rodrigues, plot_3d_mesh_comparison
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='evaluation_log.txt', filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def average_vertex_error(pred_vertices, gt_vertices):
    error = np.linalg.norm(pred_vertices - gt_vertices, axis=0)
    return np.mean(error)

def average_joint_localization_error(pred_joints, gt_joints):
    logging.info(f"pred_joints shape: {pred_joints.shape}, gt_joints shape: {gt_joints.shape}")
    if pred_joints.shape != gt_joints.shape:
        logging.error(f"Shape mismatch: pred_joints shape: {pred_joints.shape}, gt_joints shape: {gt_joints.shape}")
        return np.nan
    
    error = np.linalg.norm(pred_joints - gt_joints, axis=0)
    ave_error = np.mean(error)
    return ave_error

def average_joint_rotation_error(pred_rotations, gt_rotations):
    if pred_rotations.ndim == 1:
        pred_rotations = pred_rotations.reshape(-1, 3)
    if gt_rotations.ndim == 1:
        gt_rotations = gt_rotations.reshape(-1, 3)
    if pred_rotations.shape != gt_rotations.shape:
        logging.error(f"Shape mismatch: pred_rotations shape: {pred_rotations.shape}, gt_rotations shape: {gt_rotations.shape}")
        return np.nan
    error = np.linalg.norm(pred_rotations - gt_rotations, axis=0)
    ave_error = np.mean(error)
    return ave_error

def calculate_mesh_localization_error(pred_root_joints, gt_root_joints):
    if pred_root_joints.ndim == 1:
        pred_root_joints = pred_root_joints.reshape(-1, 3)
    if gt_root_joints.ndim == 1:
        gt_root_joints = gt_root_joints.reshape(-1, 3)
    if pred_root_joints.shape != gt_root_joints.shape:
        logging.error(f"Shape mismatch: pred_root_joints shape: {pred_root_joints.shape}, gt_root_joints shape: {gt_root_joints.shape}")
        return np.nan
    error = np.linalg.norm(pred_root_joints - gt_root_joints, axis=0)
    return np.mean(error)

def mean_per_vertex_error(pred_vertices, gt_vertices):
    error = np.linalg.norm(pred_vertices - gt_vertices, axis=1)
    mean_error = np.mean(error)
    return mean_error

def mean_per_joint_position_error(pred_joints, gt_joints):
    common_joints = min(pred_joints.shape[1], gt_joints.shape[1])
    error = np.linalg.norm(pred_joints[:common_joints, :] - gt_joints[:common_joints, :], axis=0)
    mean_error = np.mean(error)
    return mean_error

def procrustes_analysis_mpjpe(pred_joints, gt_joints):
    common_joints = min(pred_joints.shape[0], gt_joints.shape[0])
    pred_joints = pred_joints[:common_joints, :]
    gt_joints = gt_joints[:common_joints, :]

    # Center the data
    pred_joints -= np.mean(pred_joints, axis=0, keepdims=True)
    gt_joints -= np.mean(gt_joints, axis=0, keepdims=True)

    # Normalize the data
    pred_norm = np.linalg.norm(pred_joints, axis=0, keepdims=True)
    gt_norm = np.linalg.norm(gt_joints, axis=0, keepdims=True)
    
    # Handle zero norms to avoid division by zero
    pred_norm[pred_norm == 0] = 1
    gt_norm[gt_norm == 0] = 1
    
    pred_joints /= pred_norm
    gt_joints /= gt_norm

    # Regularize and handle numerical stability issues in SVD
    epsilon = 1e-8
    pred_joints += epsilon
    gt_joints += epsilon
    
    pred_joints_reshaped = pred_joints.reshape(-1, 3)
    gt_joints_reshaped = gt_joints.reshape(-1, 3)

    try:
        U, _, Vt = np.linalg.svd(np.dot(pred_joints_reshaped.T, gt_joints_reshaped))
        R = np.dot(U, Vt)
    except np.linalg.LinAlgError as e:
        logging.error(f"SVD did not converge: {e}")
        return np.inf

    pred_aligned = np.dot(pred_joints, R)
    error = np.linalg.norm(pred_aligned - gt_joints, axis=0)
    pa_mpjpe = np.mean(error)
    return pa_mpjpe

def calculate_and_save_metrics_per_frame(model, dataloader, save_dir='results'):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    frame_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", miniters=1, leave=True):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            input_shape = rawImage_XYZ.unsqueeze(2)
            betas = batch['betas'].cuda()
            pose_body = batch['pose_body'].cuda()
            trans = batch['trans'].cuda()
            root_orient = batch['root_orient'].cuda()
            gender = batch['gender'].cuda()

            smplx_output, pred_betas, pred_pose, pred_trans, pred_root_orient, faces, gender_pred = model(input_shape)

            for j in range(rawImage_XYZ.size(0)):
                pred_vertices = smplx_output[j].cpu().numpy()
                pred_joints = model.smplx_male().joints.detach().cpu().numpy()
                gt_vertices = batch['vertices'][j].cpu().numpy()
                gt_joints = batch['joints'][j].cpu().numpy()
                pred_rotations = pred_pose[j].cpu().numpy()
                gt_rotations = pose_body[j].cpu().numpy()
                pred_root_joints = pred_root_orient[j].cpu().numpy()
                gt_root_joints = root_orient[j].cpu().numpy()
                gt_gender = gender[j].cpu().numpy()

                # Ensure the correct shape for pred_joints and gt_joints
                pred_joints = pred_joints.reshape(-1, 3)
                gt_joints = gt_joints.reshape(-1, 3)

                common_joints = min(pred_joints.shape[0], gt_joints.shape[0])
                pred_joints = pred_joints[:common_joints]
                gt_joints = gt_joints[:common_joints]

                vertex_error = average_vertex_error(pred_vertices, gt_vertices)
                joint_localization_error = average_joint_localization_error(pred_joints, gt_joints)
                joint_rotation_error = average_joint_rotation_error(pred_rotations, gt_rotations)
                mesh_loc_error = calculate_mesh_localization_error(pred_root_joints, gt_root_joints)
                per_vertex_error = mean_per_vertex_error(pred_vertices, gt_vertices)
                per_joint_position_error = mean_per_joint_position_error(pred_joints, gt_joints)
                procrustes_analysis_error = procrustes_analysis_mpjpe(pred_joints, gt_joints)

                gender_accuracy = (gender_pred.argmax(dim=1)[j].cpu().numpy() == gt_gender).mean()

                metrics = {
                    'Average Vertex Error (V)': vertex_error,
                    'Average Joint Localization Error (S)': joint_localization_error,
                    'Average Joint Rotation Error (Q) (degrees)': joint_rotation_error,
                    'Mesh Localization Error (T)': mesh_loc_error,
                    'Mean Per Vertex Error (MPVE)': per_vertex_error,
                    'Mean Per Joint Position Error (MPJPE)': per_joint_position_error,
                    'Procrustes Analysis MPJPE (PA-MPJPE)': procrustes_analysis_error,
                    'Gender Prediction Accuracy (%)': gender_accuracy * 100
                }

                frame_save_dir = os.path.join(save_dir, f'frame_{frame_idx}')
                os.makedirs(frame_save_dir, exist_ok=True)

                with open(os.path.join(frame_save_dir, 'metrics_calculation.txt'), 'w') as f:
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value}\n")
                        logging.info(f"{metric}: {value}")

                fig, ax = plt.subplots()
                ax.barh(list(metrics.keys()), list(metrics.values()))
                ax.set_xlabel('Error/Accuracy')
                ax.set_title(f'Frame {frame_idx} Evaluation Metrics')
                plt.savefig(os.path.join(frame_save_dir, 'metrics_calculation.png'))
                plt.close()

                faces = batch['faces'][j].cpu().numpy()
                # Flatten the vertices arrays to ensure they are 1D
                gt_vertices = gt_vertices.reshape(-1, 3)
                pred_vertices = pred_vertices.reshape(-1, 3)
                plot_3d_mesh_comparison(gt_vertices, faces, pred_vertices, faces, f'Frame {frame_idx}', os.path.join(frame_save_dir, f'comparison_frame_{frame_idx}.png'))
                save_obj(pred_vertices, faces, os.path.join(frame_save_dir, f'prediction_frame_{frame_idx}.obj'))

                frame_idx += 1
def calculate_and_save_total_metrics(model, dataloader, save_dir='results'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    total_vertex_error = 0
    total_joint_localization_error = 0
    total_joint_rotation_error = 0
    total_mesh_loc_error = 0
    total_per_vertex_error = 0
    total_per_joint_position_error = 0
    total_procrustes_analysis_error = 0
    total_gender_accuracy = 0
    num_frames = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", miniters=1, leave=True):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            gender = batch['gender'].cuda().float().unsqueeze(1)
            input_shape = rawImage_XYZ.unsqueeze(2)
            betas = batch['betas'].cuda()
            pose_body = batch['pose_body'].cuda()
            trans = batch['trans'].cuda()
            root_orient = batch['root_orient'].cuda()

            smplx_output, pred_betas, pred_pose, pred_trans, pred_root_orient, pred_gender = model(input_shape, gender)

            for j in range(rawImage_XYZ.size(0)):
                smplx_layer = model.smplx_layer

                pred_root_orient_reshaped = pred_root_orient[j].reshape(1, 3)
                pred_root_orient_reshaped = batch_rodrigues(pred_root_orient_reshaped).reshape(1, 1, 3, 3)

                pred_pose_reshaped = pred_pose[j].reshape(-1, 3)
                pred_pose_reshaped = batch_rodrigues(pred_pose_reshaped).reshape(1, 21, 3, 3)

                smplx_output_single = smplx_layer(
                    betas=pred_betas[j].unsqueeze(0),
                    body_pose=pred_pose_reshaped,
                    global_orient=pred_root_orient_reshaped,
                    transl=pred_trans[j].unsqueeze(0)
                )

                pred_vertices = smplx_output_single.vertices.cpu().numpy().squeeze()
                pred_joints = smplx_output_single.joints.cpu().numpy().squeeze()

                gt_vertices = batch['vertices'][j].cpu().numpy()
                gt_joints = batch['joints'][j].cpu().numpy()
                pred_rotations = pred_pose[j].cpu().numpy()
                gt_rotations = batch['pose_body'][j].cpu().numpy()
                pred_root_joints = pred_root_orient[j].cpu().numpy()
                gt_root_joints = batch['root_orient'][j].cpu().numpy()
                gt_gender = gender[j].cpu().numpy()

                total_vertex_error += average_vertex_error(pred_vertices, gt_vertices)
                total_joint_localization_error += average_joint_localization_error(pred_joints, gt_joints)
                total_joint_rotation_error += average_joint_rotation_error(pred_rotations, gt_rotations)
                total_mesh_loc_error += calculate_mesh_localization_error(pred_root_joints, gt_root_joints)
                total_per_vertex_error += mean_per_vertex_error(pred_vertices, gt_vertices)
                total_per_joint_position_error += mean_per_joint_position_error(pred_joints, gt_joints)
                total_procrustes_analysis_error += procrustes_analysis_mpjpe(pred_joints, gt_joints)
                total_gender_accuracy += (pred_gender.round().cpu().numpy() == gt_gender).mean()
                num_frames += 1

    # 计算平均误差
    avg_vertex_error = total_vertex_error / num_frames
    avg_joint_localization_error = total_joint_localization_error / num_frames
    avg_joint_rotation_error = total_joint_rotation_error / num_frames
    avg_mesh_loc_error = total_mesh_loc_error / num_frames
    avg_per_vertex_error = total_per_vertex_error / num_frames
    avg_per_joint_position_error = total_per_joint_position_error / num_frames
    avg_procrustes_analysis_error = total_procrustes_analysis_error / num_frames
    avg_gender_accuracy = (total_gender_accuracy / num_frames) * 100

    total_metrics = {
        'Average Vertex Error (V) (mm)': avg_vertex_error,
        'Average Joint Localization Error (S) (mm)': avg_joint_localization_error,
        'Average Joint Rotation Error (Q) (degrees)': avg_joint_rotation_error,
        'Mesh Localization Error (T) (mm)': avg_mesh_loc_error,
        'Mean Per Vertex Error (MPVE) (mm)': avg_per_vertex_error,
        'Mean Per Joint Position Error (MPJPE) (mm)': avg_per_joint_position_error,
        'Procrustes Analysis MPJPE (PA-MPJPE) (mm)': avg_procrustes_analysis_error,
        'Gender Prediction Accuracy (%)': avg_gender_accuracy
    }

    with open(os.path.join(save_dir, 'total_metrics_calculation.txt'), 'w') as f:
        for metric, value in total_metrics.items():
            f.write(f"{metric}: {value}\n")
            logging.info(f"{metric}: {value}")

    fig, ax = plt.subplots()
    ax.barh(list(total_metrics.keys()), list(total_metrics.values()))
    ax.set_xlabel('Error/Accuracy')
    ax.set_title('Total Evaluation Metrics')
    plt.savefig(os.path.join(save_dir, 'total_metrics_calculation.png'))
    plt.close()

    return total_metrics