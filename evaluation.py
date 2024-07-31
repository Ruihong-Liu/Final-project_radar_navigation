import torch
from Main85 import create_test_loader, Simple3DConvModel, plot_3d_mesh_comparison, save_obj
from metric_calculation import calculate_and_save_metrics_per_frame,calculate_and_save_total_metrics

def evaluate_model(model_path, smplx_model_paths, input_channels, root_dir='DataUsing1', save_dir='results_evaluation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = Simple3DConvModel(smplx_model_paths['male'], smplx_model_paths['female'], input_channels).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 来跳过不匹配的层
    model.to(device)

    # 使用 Main85 中的 create_test_loader 进行评估
    test_loader = create_test_loader(root_dir)
    calculate_and_save_metrics_per_frame(model, test_loader, save_dir=save_dir)  # 使用逐帧计算函数
    # 计算并保存总的评估指标
    total_metrics = calculate_and_save_total_metrics(model, test_loader)

if __name__ == '__main__':
    model_path = r'F:\code\Final-project_radar_navigation\results\saved_model\best_model.pth'  # 替换为您的模型路径
    smplx_model_paths = {
        'male': r'F:\code\Final-project_radar_navigation\models\SMPLX_MALE.npz',
        'female': r'F:\code\Final-project_radar_navigation\models\SMPLX_FEMALE.npz'
    }
    input_channels = 31  # 替换为训练时使用的输入通道数
    evaluate_model(model_path, smplx_model_paths, input_channels)
    
