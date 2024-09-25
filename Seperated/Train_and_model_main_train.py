import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
import json
from skopt import gp_minimize
from skopt.space import Real
import matplotlib
from Train_and_model_model import Simple3DConvModelWithTripleCNNFPNAndAttention
from Train_and_model_loss import combined_loss
from Train_and_model_plotting_3D_mesh import save_random_comparison_figures
from data_loader_loader_main import get_all_file_pairs,RF3DPoseDataset,ToTensor


cudnn.enabled = False
matplotlib.use('Agg')

# ====================
# Configuration Parameters
# ====================
RESULT_PATH = 'results_Final11_full_calibrated'

# SMPLX model paths
# SMPLX model paths
SMPLX_MODEL_PATHS = {
                'male': '/home/ruihong/models/SMPLX_MALE.npz',
                'female': '/home/ruihong/models/SMPLX_FEMALE.npz',
                'neutral': '/home/ruihong/models/SMPLX_NEUTRAL.npz'
    }

# Dataset and DataLoader settings
ROOT_DIR = '/mnt/data-B/Ruihong_radar/DataUsing'
BATCH_SIZE = 2
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

# Model architecture parameters
INPUT_CHANNELS = 31
FPN_OUT_CHANNELS = 256
DROPOUT_RATE = 0.3

# Training settings
NUM_EPOCHS = 20
PATIENCE = 3

# Hyperparameter search space for Bayesian optimization
SPACE = [
    Real(1e-5, 1e-3, "log-uniform", name='lr'),
    Real(0.01, 1.0, name='betas_weight'),
    Real(0.01, 1.0, name='pose_body_weight'),
    Real(0.01, 1.0, name='root_orient_weight'),
    Real(0.01, 1.0, name='trans_weight'),
    Real(0.0001, 1.0, name='vertices_weight'),
    Real(0.1, 10.0, name='gender_weight'),
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

def collate_fn(batch):
    # 过滤掉 batch 中为 None 的样本
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None  # 如果过滤后 batch 为空，返回 None

    # 确保 batch 中每一个样本的每个字段都不是 None
    for i, sample in enumerate(batch):
        for key in sample:
            if sample[key] is None:
                logging.error(f"Sample {i} in batch has None in field '{key}'. Removing sample.")
                batch[i] = None
    
    # 再次过滤掉有 None 字段的样本
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None  # 如果过滤后 batch 为空，返回 None

    return torch.utils.data.dataloader.default_collate(batch)

def train_epoch(model, dataloader, criterion, gender_criterion, optimizer):
    model.train()
    total_loss = 0
    all_train_losses = []
    for batch in tqdm(dataloader, desc="Training Epoch"):
        if batch is None:  # 跳过空批次
            continue
        rawImage_XYZ, gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices, gt_genders = (
            batch['rawImage_XYZ'].cuda(), 
            batch['betas'][:, :10].cuda(), 
            batch['pose_body'].cuda(), 
            batch['root_orient'].cuda(), 
            batch['trans'].cuda(), 
            batch['vertices'].cuda(),  # 这里使用处理后的Ground Truth
            batch['gender'].cuda()
        )

        # 如果 rawImage_XYZ 是4维的，则首先扩展为5维
        if rawImage_XYZ.dim() == 4:
            rawImage_XYZ = rawImage_XYZ.unsqueeze(2)  # 在第三维度添加一个维度，使其变为 [batch_size, 121, 1, 111, 31]

        # 调整维度顺序，使其为 [batch_size, 31, 1, 121, 111]
        rawImage_XYZ = rawImage_XYZ.permute(0, 4, 2, 1, 3) 
        
        optimizer.zero_grad()
        pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred = model(rawImage_XYZ)
        pred_vertices = model.get_smplx_output(pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred)
        
        total_loss, losses = combined_loss(
            pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices,
            gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices,  # 使用转换后的顶点作为GT
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
            
            # 如果 rawImage_XYZ 是4维的，则首先扩展为5维
            if rawImage_XYZ.dim() == 4:
                rawImage_XYZ = rawImage_XYZ.unsqueeze(2)  # 在第三维度添加一个维度
            
            # 调整维度顺序，使其为 [batch_size, 31, 1, 121, 111]
            rawImage_XYZ = rawImage_XYZ.permute(0, 4, 2, 1, 3) 
            
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
            
            # 如果 rawImage_XYZ 是4维的，则首先扩展为5维
            if rawImage_XYZ.dim() == 4:
                rawImage_XYZ = rawImage_XYZ.unsqueeze(2)  # 在第三维度添加一个维度
            
            # 调整维度顺序，使其为 [batch_size, 31, 1, 121, 111]
            rawImage_XYZ = rawImage_XYZ.permute(0, 4, 2, 1, 3)
            
            gt_betas = batch['betas'][:, :10].cuda()
            gt_pose_body = batch['pose_body'].cuda()
            gt_root_orient = batch['root_orient'].cuda()
            gt_trans = batch['trans'].cuda()
            gt_vertices = batch['vertices'].cuda()
            gt_genders = batch['gender'].cuda()

            logging.info(f"Running model forward pass for batch {batch_idx}...")

            pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred = model(rawImage_XYZ)
            pred_vertices = model.get_smplx_output(pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred)

            logging.info(f"Model forward pass complete for batch {batch_idx}.")
            
            # 随机选择样本进行绘图
            if batch_idx == 0:  # 仅在第一个batch中进行
                save_random_comparison_figures(pred_vertices, gt_vertices, model.faces_male, model.faces_female, gt_genders, iteration, result_path, phase, num_samples=5)

        logging.info(f"{phase.capitalize()} evaluation and plotting complete for iteration {iteration}.")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_fn)
    
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
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_loss_per_epoch = train_epoch(model, train_loader, criterion, gender_criterion, optimizer)
        val_loss, val_loss_per_epoch = validate_epoch(model, val_loader, criterion, gender_criterion, epoch, RESULT_PATH, bayes_iter)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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
        
        logging.info(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
        print(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Train Loss: {train_loss:.4f}')
        print(f'Epoch [{epoch+1}/{FINAL_NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
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
