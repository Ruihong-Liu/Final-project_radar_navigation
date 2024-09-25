import torch
import torch.nn as nn
import torch.nn.functional as F
from smplx import SMPLXLayer

# change the matrix to correspond form
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

# set-up a multihead attention model
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

# setting FPN with attention
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

# setting the final model for training
class Simple3DConvModelWithTripleCNNFPNAndAttention(nn.Module):
    def __init__(self, smplx_model_paths, input_channels=31, fpn_out_channels=256, reduced_channels=128, dropout_rate=0.0):
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
        return vertices*1000