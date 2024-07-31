import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data_loader import RF3DPoseDataset, ToTensor, load_mat_file, load_obj_file, analyze_dataset
from smplx import SMPLXLayer
import matplotlib.pyplot as plt
import matplotlib
import torch.optim as optim

matplotlib.use('Agg')

def configure_logging(log_file_path):
    print(f"Configuring logging. Log file path: {log_file_path}")
    if log_file_path:
        directory = os.path.dirname(log_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory for log files: {directory}")

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            handlers=[file_handler, stream_handler])

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        attention = torch.sigmoid(out)
        return x * attention

class Simple3DConvModel(nn.Module):
    def __init__(self, smplx_male_path, smplx_female_path, input_channels):
        super(Simple3DConvModel, self).__init__()
        self.layer1 = self._make_layer(input_channels, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)  # 增加一个卷积层

        self.attn1 = AttentionBlock(64, 64)
        self.attn2 = AttentionBlock(128, 128)
        self.attn3 = AttentionBlock(256, 256)
        self.attn4 = AttentionBlock(512, 512)  # 对应新的卷积层

        self.fpn1 = nn.Conv3d(512, 256, kernel_size=1)  # 更新FPN的输入通道
        self.fpn2 = nn.Conv3d(256, 256, kernel_size=1)
        self.fpn3 = nn.Conv3d(128, 256, kernel_size=1)
        self.fpn4 = nn.Conv3d(64, 256, kernel_size=1)

        self.fc1 = nn.Linear(256 * 1 * 16 * 14, 512)  # 更新全连接层的输入维度
        self.dropout1 = nn.Dropout(p=0.3)  # 添加Dropout层
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.3)  # 添加Dropout层
        self.fc3 = nn.Linear(256, 79)
        self.gender_fc = nn.Linear(256, 2)

        self.smplx_male = SMPLXLayer(model_path=smplx_male_path, gender='male', use_pca=False)
        self.smplx_female = SMPLXLayer(model_path=smplx_female_path, gender='female', use_pca=False)
        self.faces = self.smplx_male.faces

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock3D(in_channels, out_channels, stride)

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.attn1(x1)
        logging.info(f"After layer1: min={x1.min()}, max={x1.max()}, mean={x1.mean()}")

        x2 = self.layer2(x1)
        x2 = self.attn2(x2)
        logging.info(f"After layer2: min={x2.min()}, max={x2.max()}, mean={x2.mean()}")

        x3 = self.layer3(x2)
        x3 = self.attn3(x3)
        logging.info(f"After layer3: min={x3.min()}, max={x3.max()}, mean={x3.mean()}")

        x4 = self.layer4(x3)
        x4 = self.attn4(x4)
        logging.info(f"After layer4: min={x4.min()}, max={x4.max()}, mean={x4.mean()}")

        f1 = self.fpn1(x4)
        f2 = F.interpolate(self.fpn2(x3), size=f1.shape[2:], mode='trilinear', align_corners=False) + f1
        f3 = F.interpolate(self.fpn3(x2), size=f2.shape[2:], mode='trilinear', align_corners=False) + f2
        f4 = F.interpolate(self.fpn4(x1), size=f3.shape[2:], mode='trilinear', align_corners=False) + f3
        logging.info(f"After FPN and interpolation: min={f4.min()}, max={f4.max()}, mean={f4.mean()}")

        f4 = f4.view(f4.size(0), -1)
        logging.info(f"After flatten: min={f4.min()}, max={f4.max()}, mean={f4.mean()}")

        x = F.relu(self.fc1(f4))
        x = self.dropout1(x)
        logging.info(f"After fc1: min={x.min()}, max={x.max()}, mean={x.mean()}")

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logging.info(f"After fc2: min={x.min()}, max={x.max()}, mean={x.mean()}")

        x_betas_pose = self.fc3(x)
        gender_pred = self.gender_fc(x)
        logging.info(f"After fc3 (x_betas_pose): min={x_betas_pose.min()}, max={x_betas_pose.max()}, mean={x_betas_pose.mean()}")

        pred_betas = x_betas_pose[:, :10]
        pred_pose = x_betas_pose[:, 10:73]
        pred_trans = x_betas_pose[:, 73:76]
        pred_root_orient = x_betas_pose[:, 76:79]

        pred_betas = pred_betas[:, :10]
        pred_root_orient_mat = batch_rodrigues(pred_root_orient.reshape(-1, 3)).reshape(-1, 3, 3)
        pred_pose_mat = batch_rodrigues(pred_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3)

        gender_pred_labels = gender_pred.argmax(dim=1)
        smplx_output = self._select_smplx_layer(pred_betas, pred_pose_mat, pred_root_orient_mat, pred_trans, gender_pred_labels)
        logging.info(f"SMPLX output: min={smplx_output.min()}, max={smplx_output.max()}, mean={smplx_output.mean()}")

        return smplx_output, pred_betas, pred_pose, pred_trans, pred_root_orient, self.faces, gender_pred


    def _select_smplx_layer(self, betas, pose_mat, root_orient_mat, trans, gender_pred_labels):
        smplx_output_male = self.smplx_male(betas=betas, body_pose=pose_mat, global_orient=root_orient_mat, transl=trans)
        smplx_output_female = self.smplx_female(betas=betas, body_pose=pose_mat, global_orient=root_orient_mat, transl=trans)

        smplx_output = torch.where(gender_pred_labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) == 0,
                                   smplx_output_male.vertices,
                                   smplx_output_female.vertices)

        return smplx_output

def batch_rodrigues(rot_vecs):
    batch_size = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    axis = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros(batch_size, 1, device=rot_vecs.device)
    K = torch.cat([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=1).view(batch_size, 3, 3)

    I = torch.eye(3, device=rot_vecs.device).unsqueeze(0)
    rot_mats = cos * I + (1 - cos) * torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1)) + sin * K
    return rot_mats

def calculate_accuracy(predictions, targets, threshold=0.1):
    absolute_errors = torch.abs(predictions - targets)
    correct_predictions = (absolute_errors < threshold).float()
    accuracy = correct_predictions.mean().item() * 100
    return accuracy

def save_model(model, epoch, save_dir='results/saved_model'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig('results/metrics.png')
    plt.close()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.access(directory, os.W_OK):
        print(f"Directory {directory} is writable.")
    else:
        print(f"Directory {directory} is not writable.")

def save_obj(vertices, faces, file_path):
    try:
        if vertices is None or faces is None:
            logging.error(f"Vertices or faces are None. Skipping saving OBJ to {file_path}.")
            return

        if len(vertices) == 0 or len(faces) == 0:
            logging.error(f"Vertices or faces are empty. Skipping saving OBJ to {file_path}.")
            return

        with open(file_path, 'w') as f:
            for vertex in vertices:
                for v in vertex:  # 确保每个顶点是一个包含三个浮点数的列表或数组
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                if isinstance(face, (np.ndarray, list)) and len(face) == 3:
                    f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                else:
                    logging.error(f"Invalid face format: {face}")
        logging.info(f"Saved OBJ to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save OBJ to {file_path}: {e}")


def plot_3d_mesh_comparison(vertices_gt, faces_gt, vertices_pred, faces_pred, title, save_path):
    try:
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(vertices_gt[:, 0], vertices_gt[:, 1], vertices_gt[:, 2], triangles=faces_gt, cmap='viridis', edgecolor='none')
        ax1.set_title('Ground Truth')

        ax2 = fig.add_subplot(122, projection='3d')
        vertices_pred = vertices_pred.reshape(-1, vertices_pred.shape[-1])
        faces_pred = faces_pred.reshape(-1, faces_pred.shape[-1])
        ax2.plot_trisurf(vertices_pred[:, 0], vertices_pred[:, 1], vertices_pred[:, 2], triangles=faces_pred, cmap='viridis', edgecolor='none')
        ax2.set_title('Prediction')

        plt.suptitle(title)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot 3D mesh comparison: {e}")

def train_epoch(model, dataloader, criterion, gender_criterion, optimizer, gender_weight=0.001, clip_value=1.0):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    for batch in tqdm(dataloader, desc="Training", miniters=1, leave=True):
        rawImage_XYZ = batch['rawImage_XYZ'].cuda()
        input_shape = rawImage_XYZ.unsqueeze(2)

        betas = batch['betas'].cuda()
        pose_body = batch['pose_body'].cuda()
        trans = batch['trans'].cuda()
        root_orient = batch['root_orient'].cuda()
        gender = batch['gender'].cuda()

        optimizer.zero_grad()
        try:
            smplx_output, pred_betas, pred_pose, pred_trans, pred_root_orient, faces, gender_pred = model(input_shape)

            loss_betas = criterion(pred_betas, betas[:, :10])
            loss_pose = criterion(pred_pose, pose_body)
            loss_trans = criterion(pred_trans, trans)
            loss_root_orient = criterion(pred_root_orient, root_orient)
            loss_gender = gender_criterion(gender_pred, gender)

            loss = loss_betas + loss_pose + loss_trans + loss_root_orient + gender_weight * loss_gender

            if torch.isnan(loss) or torch.isinf(loss):
                logging.error("NaN or Inf loss encountered. Skipping this batch.")
                continue

            logging.info(f"Loss: {loss.item()}")

            accuracy_betas = calculate_accuracy(pred_betas, betas[:, :10])
            accuracy_pose = calculate_accuracy(pred_pose, pose_body)
            accuracy_trans = calculate_accuracy(pred_trans, trans)
            accuracy_root_orient = calculate_accuracy(pred_root_orient, root_orient)
            accuracy_gender = (gender_pred.argmax(dim=1) == gender).float().mean().item() * 100
            accuracy = (accuracy_betas + accuracy_pose + accuracy_trans + accuracy_root_orient + accuracy_gender) / 5

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy
        except Exception as e:
            logging.error(f"Error during training batch: {e}")
            continue

    avg_loss = epoch_loss / len(dataloader)
    avg_accuracy = epoch_accuracy / len(dataloader)
    return avg_loss, avg_accuracy



def validate_epoch(model, val_loader, criterion, gender_criterion, epoch, gender_weight=0.001):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    save_dir = 'results/validation'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=True):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            betas = batch['betas'].cuda()
            pose_body = batch['pose_body'].cuda()
            trans = batch['trans'].cuda()
            root_orient = batch['root_orient'].cuda()
            gender = batch['gender'].cuda()

            input_shape = rawImage_XYZ.unsqueeze(2)

            try:
                smplx_output, pred_betas, pred_pose, pred_trans, pred_root_orient, faces, gender_pred = model(input_shape)

                loss_betas = criterion(pred_betas, betas[:, :10])
                loss_pose = criterion(pred_pose, pose_body)
                loss_trans = criterion(pred_trans, trans)
                loss_root_orient = criterion(pred_root_orient, root_orient)
                loss_gender = gender_criterion(gender_pred, gender)

                loss = loss_betas + loss_pose + loss_trans + loss_root_orient + gender_weight * loss_gender

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error("NaN or Inf loss encountered. Skipping this batch.")
                    continue

                accuracy_betas = calculate_accuracy(pred_betas, betas[:, :10])
                accuracy_pose = calculate_accuracy(pred_pose, pose_body)
                accuracy_trans = calculate_accuracy(pred_trans, trans)
                accuracy_root_orient = calculate_accuracy(pred_root_orient, root_orient)
                accuracy_gender = (gender_pred.argmax(dim=1) == gender).float().mean().item() * 100
                accuracy = (accuracy_betas + accuracy_pose + accuracy_trans + accuracy_root_orient + accuracy_gender) / 5

                val_loss += loss.item()
                correct += accuracy
                total += 1
            except Exception as e:
                logging.error(f"Error during validation batch: {e}")
                continue

    avg_loss = val_loss / total
    avg_accuracy = correct / total
    return avg_loss, avg_accuracy


def test_model(model, test_loader, criterion, gender_criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    save_dir = 'results/testing'

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            rawImage_XYZ = batch['rawImage_XYZ'].cuda()
            betas = batch['betas'].cuda()
            pose_body = batch['pose_body'].cuda()
            trans = batch['trans'].cuda()
            root_orient = batch['root_orient'].cuda()
            gender = batch['gender'].cuda()

            input_shape = rawImage_XYZ.unsqueeze(2)

            smplx_output, pred_betas, pred_pose, pred_trans, pred_root_orient, faces, gender_pred = model(input_shape)
            loss_betas = criterion(pred_betas, betas[:, :10])
            loss_pose = criterion(pred_pose, pose_body)
            loss_trans = criterion(pred_trans, trans)
            loss_root_orient = criterion(pred_root_orient, root_orient)
            loss_gender = gender_criterion(gender_pred, gender)

            loss = loss_betas + loss_pose + loss_trans + loss_root_orient + loss_gender

            accuracy_betas = calculate_accuracy(pred_betas, betas[:, :10])
            accuracy_pose = calculate_accuracy(pred_pose, pose_body)
            accuracy_trans = calculate_accuracy(pred_trans, trans)
            accuracy_root_orient = calculate_accuracy(pred_root_orient, root_orient)
            accuracy_gender = (gender_pred.argmax(dim=1) == gender).float().mean().item() * 100
            accuracy = (accuracy_betas + accuracy_pose + accuracy_trans + accuracy_root_orient + accuracy_gender) / 5

            test_loss += loss.item()
            correct += accuracy
            total += 1

            if batch_idx < 5:
                vertices_gt, faces_gt = batch['vertices'][0].cpu().numpy(), batch['faces'][0].cpu().numpy()
                vertices_pred = smplx_output[0, 0].cpu().numpy()  # 选择第一个样本的预测数据
                faces_pred = faces

                plot_3d_mesh_comparison(vertices_gt, faces_gt, vertices_pred, faces_pred, f'Test Batch {batch_idx}', os.path.join(save_dir, f'comparison_test_{batch_idx}.png'))
                save_obj(vertices_pred, faces_pred, os.path.join(save_dir, f'prediction_test_{batch_idx}.obj'))

    avg_loss = test_loss / total
    avg_accuracy = correct / total

def main_train():
    # 初始化日志
    configure_logging('results/main_train_log.txt')
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)

    ensure_directory_exists('results')

    root_dir = 'DataUsing'
    batch_size = 32
    num_epochs = 30
    gender_weight = 0.00001
    learning_rate = 0.00001  # 调整学习率

    weight_decay = 1e-4
    smplx_male_path = r'F:\code\Final-project_radar_navigation\models\SMPLX_MALE.npz'
    smplx_female_path = r'F:\code\Final-project_radar_navigation\models\SMPLX_FEMALE.npz'

    matched_data = analyze_dataset(root_dir)
    dataset = RF3DPoseDataset(matched_data, transform=ToTensor())
    main_logger.info(f"Total dataset size: {len(dataset)}")

    if len(dataset) == 0:
        main_logger.error("Dataset is empty after filtering. Exiting.")
        return

    sample_item = dataset[0]
    main_logger.info(f"Sample rawImage_XYZ shape: {sample_item['rawImage_XYZ'].shape}")

    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    main_logger.info(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    if train_size == 0 or val_size == 0 or test_size == 0:
        main_logger.error("One of the dataset splits is zero. Exiting.")
        return

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # 在数据加载时检查数据的数值范围
    for batch in train_loader:
        rawImage_XYZ = batch['rawImage_XYZ']
        logging.info(f"rawImage_XYZ stats: min={rawImage_XYZ.min()}, max={rawImage_XYZ.max()}, mean={rawImage_XYZ.mean()}")
        
        betas = batch['betas']
        pose_body = batch['pose_body']
        trans = batch['trans']
        root_orient = batch['root_orient']
        
        logging.info(f"betas stats: min={betas.min()}, max={betas.max()}, mean={betas.mean()}")
        logging.info(f"pose_body stats: min={pose_body.min()}, max={pose_body.max()}, mean={pose_body.mean()}")
        logging.info(f"trans stats: min={trans.min()}, max={trans.max()}, mean={trans.mean()}")
        logging.info(f"root_orient stats: min={root_orient.min()}, max={root_orient.max()}, mean={root_orient.mean()}")
        break  # 只检查一个批次

    input_channels = sample_item['rawImage_XYZ'].shape[0]

    model = Simple3DConvModel(smplx_male_path, smplx_female_path, input_channels).cuda()
    model.apply(weights_init)
    criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_path = os.path.join('results', 'saved_model', 'best_model.pth')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(num_epochs):
        main_logger.info(f"Starting epoch {epoch+1}/{num_epochs}...")
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, gender_criterion, optimizer, gender_weight)
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, gender_criterion, epoch, gender_weight)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        main_logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        save_model(model, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            main_logger.info(f"New best model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    model.load_state_dict(torch.load(best_model_path))
    test_model(model, test_loader, criterion, gender_criterion)

if __name__ == "__main__":
    main_train()