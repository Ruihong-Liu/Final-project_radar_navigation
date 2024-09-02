import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from data_loader import RF3DPoseDataset, ToTensor, get_all_file_pairs
from smplx import SMPLXLayer
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.backends.cudnn as cudnn
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torchvision import transforms
import random
import json
from skopt import gp_minimize
from skopt.space import Real
import matplotlib

cudnn.enabled = False
matplotlib.use('Agg')

# ====================
# Configuration Parameters
# ====================
RESULT_PATH = 'results_Final11_multi_head'

# SMPLX model paths
SMPLX_MODEL_PATHS = {
                'male': '/home/ruihong/models/SMPLX_MALE.npz',
                'female': '/home/ruihong/models/SMPLX_FEMALE.npz',
                'neutral': '/home/ruihong/models/SMPLX_NEUTRAL.npz'
    }

# Dataset and DataLoader settings
ROOT_DIR = '/mnt/data-B/Ruihong_radar/DataUsing1'
BATCH_SIZE = 4
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

# Model architecture parameters
INPUT_CHANNELS = 31
FPN_OUT_CHANNELS = 256
DROPOUT_RATE = 0.4

# Training settings
NUM_EPOCHS = 20
PATIENCE = 3

# Hyperparameter search space for Bayesian optimization
# 修改搜索空间，确保权重不低于 1.0
SPACE = [
    Real(1e-5, 1e-3, "log-uniform", name='lr'),
    Real(3.0, 10.0, name='betas_weight'),        # 最小为 1.0
    Real(3.0, 10.0, name='pose_body_weight'),    # 最小为 1.0
    Real(1.0, 10.0, name='root_orient_weight'),  # 最小为 1.0
    Real(1.0, 10.0, name='trans_weight'),        # 最小为 1.0
    Real(1.0, 10.0, name='vertices_weight'),     # 最小为 1.0
    Real(1.0, 10.0, name='gender_weight'),       # 最小为 1.0
    Real(1e-6, 1e-2, "log-uniform", name='l2_lambda')
]

# Final training settings
FINAL_NUM_EPOCHS = 50
FINAL_PATIENCE = 5

# ====================
# Logging setup
# ====================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training_log.txt',
                    filemode='w')

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_human_mesh(vertices, faces, filename):
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
    except Exception as e:
        print(f"Error saving mesh to {filename}: {e}")
        print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")

def plot_mesh(ax, vertices, faces, title):
    ax.set_title(title)
    mesh = Poly3DCollection(vertices[faces], alpha=0.1, edgecolor='k')
    ax.add_collection3d(mesh)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def save_random_comparison_figures(pred_vertices, gt_vertices, faces_male, faces_female, genders, iteration, result_path, phase, num_samples=5):
    num_samples = min(num_samples, pred_vertices.size(0))  # 确保样本数量不超过实际的batch大小
    indices = random.sample(range(pred_vertices.size(0)), num_samples)

    for i in indices:
        filename = f"{result_path}/{phase}_comparison_iter_{iteration}_sample_{i}.png"
        plot_comparison(pred_vertices[i].cpu().numpy(), gt_vertices[i].cpu().numpy(), faces_male, faces_female, genders.cpu().numpy(), i, filename)
        print(f"Saved {phase} comparison figure: {filename}")

def plot_comparison(pred_vertices, gt_vertices, faces_male, faces_female, genders, idx, filename):
    fig = plt.figure(figsize=(12, 8))

    # Plot predictions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_mesh(ax1, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Top View')
    ax1.view_init(elev=90, azim=-90)  # 设置为顶视图

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_mesh(ax2, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Side View')
    ax2.view_init(elev=0, azim=-90)  # 设置为前视图

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_mesh(ax3, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Front View')
    ax3.view_init(elev=0, azim=0)  # 设置为侧视图

    # Plot ground truth
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_mesh(ax4, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Top View')
    ax4.view_init(elev=90, azim=-90)  # 设置为顶视图

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    plot_mesh(ax5, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Side View')
    ax5.view_init(elev=0, azim=-90)  # 设置为前视图

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_mesh(ax6, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Front View')
    ax6.view_init(elev=0, azim=0)  # 设置为侧视图

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def batch_rodrigues(theta):
    theta = theta.contiguous().view(-1, 3)
    angle = torch.norm(theta, dim=1, keepdim=True)
    r = theta / (angle + 1e-8)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    r_x, r_y, r_z = r[:, 0], r[:, 1], r[:, 2]

    K = torch.stack([
        torch.zeros_like(r_x), -r_z, r_y,
        r_z, torch.zeros_like(r_x), -r_x,
        -r_y, r_x, torch.zeros_like(r_x)
    ], dim=1).view(-1, 3, 3)

    I = torch.eye(3, device=theta.device).unsqueeze(0)

    rotmat = I + sin.view(-1, 1, 1) * K + (1 - cos.view(-1, 1, 1)) * torch.bmm(K, K)

    return rotmat

def quat_to_rotmat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    rotmat = torch.stack(
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
        dim=1).view(B, 3, 3)
    return rotmat

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttentionModule, self).__init__()
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()

        # Apply the convolutional layers and split into heads
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [batch_size, num_heads, depth*height*width, head_dim]
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)  # [batch_size, num_heads, head_dim, depth*height*width]
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [batch_size, num_heads, depth*height*width, head_dim]

        # Attention score calculation
        energy = torch.matmul(query, key) / (self.head_dim ** 0.5)  # [batch_size, num_heads, depth*height*width, depth*height*width]
        attention = F.softmax(energy, dim=-1)  # [batch_size, num_heads, depth*height*width, depth*height*width]

        # Attention application
        out = torch.matmul(attention, value)  # [batch_size, num_heads, depth*height*width, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous()  # [batch_size, num_heads, head_dim, depth*height*width]
        out = out.view(batch_size, C, depth, height, width)  # [batch_size, C, depth, height, width]
        
        # Output
        out = self.gamma * out + x
        return out

class FPN3DWithMultiHeadAttention(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_heads=8):
        super(FPN3DWithMultiHeadAttention, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.attention_modules = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            output_conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            attention_module = MultiHeadAttentionModule(out_channels, num_heads)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
            self.attention_modules.append(attention_module)

    def forward(self, inputs):
        # 生成侧分支
        laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]

        # 从高分辨率到低分辨率融合特征图
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样和裁剪
            upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="trilinear", align_corners=False)
            laterals[i] += upsampled

        # 对每个侧分支应用卷积和多头注意力机制
        outputs = [attention_module(output_conv(lateral)) 
                   for output_conv, attention_module, lateral in zip(self.output_convs, self.attention_modules, laterals)]
        return outputs


class Simple3DConvModelWithTripleCNNFPNAndAttention(nn.Module):
    def __init__(self, smplx_model_paths, input_channels=31, fpn_out_channels=256, reduced_channels=128, dropout_rate=0.2):
        super(Simple3DConvModelWithTripleCNNFPNAndAttention, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fpn = FPN3DWithMultiHeadAttention([64, 128, 256], fpn_out_channels)

        self.conv1x1 = nn.Conv3d(fpn_out_channels, reduced_channels, kernel_size=1)

        # 重新计算展平后的尺寸：128 * 1 * 16 * 14
        flattened_size = reduced_channels * 1 * 16 * 14

        # 更新全连接层的输入尺寸
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc2_betas = nn.Linear(1024, 10)
        self.fc2_pose_body = nn.Linear(1024, 63)
        self.fc2_root_orient = nn.Linear(1024, 3)
        self.fc2_trans = nn.Linear(1024, 3)
        self.fc2_gender = nn.Linear(1024, 1)

        self.smplx_layer_male = SMPLXLayer(model_path=smplx_model_paths['male'], gender='male', use_pca=False)
        self.smplx_layer_female = SMPLXLayer(model_path=smplx_model_paths['female'], gender='female', use_pca=False)
        self.smplx_layer_neutral = SMPLXLayer(model_path=smplx_model_paths['neutral'], gender='neutral', use_pca=False)
        
        self.faces_male = self.smplx_layer_male.faces
        self.faces_female = self.smplx_layer_female.faces
        self.faces_neutral = self.smplx_layer_neutral.faces

        self.loss_weights = {
            'betas': 2.0,
            'pose_body': 5.0,
            'root_orient': 1.0,
            'trans': 1.0,
            'vertices': 0.001,
            'gender': 1.0
        }

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))

        fpn_outs = self.fpn([c1, c2, c3])

        x = F.relu(self.conv1x1(fpn_outs[-1]))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        pred_betas = self.fc2_betas(x)
        pred_pose_body = self.fc2_pose_body(x)
        pred_root_orient = self.fc2_root_orient(x)
        pred_trans = self.fc2_trans(x)
        gender_pred = torch.sigmoid(self.fc2_gender(x))
        
        return pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred

    def get_smplx_output(self, pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender):
        vertices_list = []
        for i in range(gender.shape[0]):
            gender_value = gender[i].item()
            if gender_value > 0.5:
                smplx_layer = self.smplx_layer_female
            else:
                smplx_layer = self.smplx_layer_male
            
            pred_root_orient_matrix = batch_rodrigues(pred_root_orient[i].unsqueeze(0)).unsqueeze(0)
            pred_pose_body_matrix = batch_rodrigues(pred_pose_body[i].view(-1, 3)).view(-1, 21, 3, 3)
            
            output = smplx_layer(
                betas=pred_betas[i].unsqueeze(0), 
                body_pose=pred_pose_body_matrix, 
                global_orient=pred_root_orient_matrix, 
                transl=pred_trans[i].unsqueeze(0)
            )
            vertices_list.append(output.vertices)
        vertices = torch.cat(vertices_list, dim=0)
        return vertices

def combined_loss(pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices, 
                  gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices, 
                  pred_genders, gt_genders, criterion, gender_criterion, model):
    
    betas_loss = criterion(pred_betas, gt_betas)
    pose_body_loss = criterion(pred_pose_body, gt_pose_body)
    root_orient_loss = criterion(pred_root_orient, gt_root_orient)
    trans_loss = criterion(pred_trans, gt_trans)
    vertices_loss = criterion(pred_vertices, gt_vertices)
    gender_loss = gender_criterion(pred_genders, gt_genders.unsqueeze(1).float())
    
    losses = {
        'betas': betas_loss,
        'pose_body': pose_body_loss,
        'root_orient': root_orient_loss,
        'trans': trans_loss,
        'vertices': vertices_loss,
        'gender': gender_loss
    }

    total_loss = sum(model.loss_weights[key] * loss for key, loss in losses.items())
    
    return total_loss, losses

def train_epoch(model, dataloader, criterion, gender_criterion, optimizer):
    model.train()
    total_loss = 0
    all_train_losses = []
    for batch in tqdm(dataloader, desc="Training Epoch"):
        rawImage_XYZ, gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices, gt_genders = (
            batch['rawImage_XYZ'].cuda(), 
            batch['betas'][:, :10].cuda(), 
            batch['pose_body'].cuda(), 
            batch['root_orient'].cuda(), 
            batch['trans'].cuda(), 
            batch['vertices'].cuda(), 
            batch['gender'].cuda()
        )
        
        rawImage_XYZ = rawImage_XYZ.unsqueeze(2)
        
        optimizer.zero_grad()
        pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred = model(rawImage_XYZ)
        pred_vertices = model.get_smplx_output(pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred)
        
        total_loss, losses = combined_loss(
            pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices,
            gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices,
            gender_pred, gt_genders, criterion, gender_criterion, model
        )
        
        total_loss.backward()
        optimizer.step()
        total_loss_value = total_loss.item()
        all_train_losses.append(total_loss_value)
    return sum(all_train_losses) / len(all_train_losses), all_train_losses

def validate_epoch(model, dataloader, criterion, gender_criterion, epoch, result_path, bayes_iter):
    model.eval()
    all_val_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            if rawImage_XYZ.dim() == 4:
                rawImage_XYZ = rawImage_XYZ.unsqueeze(2)
            gt_betas = batch['betas'][:, :10].cuda()
            gt_pose_body = batch['pose_body'].cuda()
            gt_root_orient = batch['root_orient'].cuda()
            gt_trans = batch['trans'].cuda()
            gt_vertices = batch['vertices'].cuda()
            gt_genders = batch['gender'].cuda()

            pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred = model(rawImage_XYZ)
            pred_vertices = model.get_smplx_output(pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred)

            total_loss, losses = combined_loss(
                pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices,
                gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices,
                gender_pred, gt_genders,
                criterion, gender_criterion, model
            )
            all_val_losses.append(total_loss.item())

            if batch_idx == 0:  # 仅在第一个batch中进行
                save_random_comparison_figures(pred_vertices, gt_vertices, model.faces_male, model.faces_female, gt_genders, epoch, result_path, bayes_iter, num_samples=5)

    avg_val_loss = sum(all_val_losses) / len(all_val_losses)
    return avg_val_loss, all_val_losses

def evaluate_and_plot(model, dataloader, iteration, result_path, phase="test"):
    print(f"Starting {phase} evaluation and plotting for iteration {iteration}...")

    model.eval()
    ensure_directory_exists(result_path)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            if rawImage_XYZ.dim() == 4:
                rawImage_XYZ = rawImage_XYZ.unsqueeze(2)
            gt_betas = batch['betas'][:, :10].cuda()
            gt_pose_body = batch['pose_body'].cuda()
            gt_root_orient = batch['root_orient'].cuda()
            gt_trans = batch['trans'].cuda()
            gt_vertices = batch['vertices'].cuda()
            gt_genders = batch['gender'].cuda()

            print(f"Running model forward pass for batch {batch_idx}...")

            pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred = model(rawImage_XYZ)
            pred_vertices = model.get_smplx_output(pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred)

            print(f"Model forward pass complete for batch {batch_idx}.")

            # 随机选择样本进行绘图
            if batch_idx == 0:  # 仅在第一个batch中进行
                save_random_comparison_figures(pred_vertices, gt_vertices, model.faces_male, model.faces_female, gt_genders, iteration, result_path, phase, num_samples=5)

        print(f"{phase.capitalize()} evaluation and plotting complete for iteration {iteration}.")

def objective(params, bayes_iter):
    train_losses = []
    val_losses = []
    
    lr, betas_weight, pose_body_weight, root_orient_weight, trans_weight, vertices_weight, gender_weight, l2_lambda = params
    
    file_pairs = get_all_file_pairs(ROOT_DIR)
    dataset = RF3DPoseDataset(file_pairs, transform=transforms.Compose([ToTensor()]))
    
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = int(VAL_RATIO * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Simple3DConvModelWithTripleCNNFPNAndAttention(smplx_model_paths=SMPLX_MODEL_PATHS).cuda()
    criterion = nn.L1Loss()
    gender_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

    model.loss_weights = {
        'betas': betas_weight,
        'pose_body': pose_body_weight,
        'root_orient': root_orient_weight,
        'trans': trans_weight,
        'vertices': vertices_weight,
        'gender': gender_weight
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # 定义最佳模型保存路径
    model_save_dir = os.path.join(RESULT_PATH, 'model')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_loss_per_epoch = train_epoch(model, train_loader, criterion, gender_criterion, optimizer)
        val_loss, val_loss_per_epoch = validate_epoch(model, val_loader, criterion, gender_criterion, epoch, RESULT_PATH, bayes_iter)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        
        # Check if the current model is the best one so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(model_save_dir, f'best_model_iter_{bayes_iter}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at: {best_model_path}")
            logging.info(f"Best model saved at: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                logging.info("Early stopping triggered.")
                break

    # 在每次贝叶斯优化的iteration后进行测试并保存对比图
    evaluate_and_plot(model, test_loader, bayes_iter, RESULT_PATH, phase="test")

    # 绘制并保存损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Bayesian Optimization Iteration {bayes_iter}: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    save_path = f'{RESULT_PATH}/loss_plot_iter_{bayes_iter}.png'
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved loss plot for iteration {bayes_iter} at {save_path}")

    return best_val_loss


def final_training(best_params):
    file_pairs = get_all_file_pairs(ROOT_DIR)
    dataset = RF3DPoseDataset(file_pairs, transform=transforms.Compose([ToTensor()]))
    
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = int(VAL_RATIO * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Simple3DConvModelWithTripleCNNFPNAndAttention(smplx_model_paths=SMPLX_MODEL_PATHS).cuda()
    criterion = nn.L1Loss()
    gender_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['l2_lambda'])

    model.loss_weights = {
        'betas': best_params['betas_weight'],
        'pose_body': best_params['pose_body_weight'],
        'root_orient': best_params['root_orient_weight'],
        'trans': best_params['trans_weight'],
        'vertices': best_params['vertices_weight'],
        'gender': best_params['gender_weight']
    }

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(FINAL_NUM_EPOCHS):
        train_loss, train_loss_per_epoch = train_epoch(model, train_loader, criterion, gender_criterion, optimizer)
        val_loss, val_loss_per_epoch = validate_epoch(model, val_loader, criterion, gender_criterion, epoch, RESULT_PATH, 'final_training')
        
        print(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        print(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FINAL_PATIENCE:
                print("Early stopping triggered.")
                logging.info("Early stopping triggered.")
                break

    torch.save(model.state_dict(), 'final_trained_model.pth')
    print("Final model saved to 'final_trained_model.pth'")

if __name__ == '__main__':
    ensure_directory_exists(RESULT_PATH)

    def wrapped_objective(params):
        wrapped_objective.iter_count += 1
        return objective(params, wrapped_objective.iter_count)

    wrapped_objective.iter_count = 0

    res = gp_minimize(wrapped_objective, SPACE, n_calls=50, random_state=0)

    best_params = {
        'lr': res.x[0],
        'betas_weight': res.x[1],
        'pose_body_weight': res.x[2],
        'root_orient_weight': res.x[3],
        'trans_weight': res.x[4],
        'vertices_weight': res.x[5],
        'gender_weight': res.x[6],
        'l2_lambda': res.x[7]
    }

    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f)

    print(f"Best parameters saved to 'best_hyperparameters.json'")

    with open('best_hyperparameters.json', 'r') as f:
        best_params = json.load(f)

    final_training(best_params)
