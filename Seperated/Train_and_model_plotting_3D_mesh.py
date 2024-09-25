import matplotlib
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import logging
import matplotlib.pyplot as plt
import numpy as np

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
    
    # 动态计算中心点
    def calculate_limits(vertices):
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        center = (min_vals + max_vals) / 2
        range_vals = max_vals - min_vals
        max_range = range_vals.max() / 2
        return center, max_range

    pred_center, pred_max_range = calculate_limits(pred_vertices)
    gt_center, gt_max_range = calculate_limits(gt_vertices)
    
    # 使用顶点计算出的最大范围设置坐标轴
    max_range = max(pred_max_range, gt_max_range)
    top_lim = [-max_range, max_range]
    side_front_lim = [-max_range, max_range]

    def set_axes_limits(ax, center, max_range):
        # 记录 center 和 max_range 的值以便调试
        logging.info(f"设置坐标轴限制，center: {center}, max_range: {max_range}")
        
        # 检查输入中是否有 NaN 或 Inf 值，如果有则替换为默认值
        if np.any(np.isnan(center)) or np.any(np.isinf(center)):
            logging.warning("检测到无效的 center 值，将 NaN/Inf 替换为 0。")
            center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(max_range) or np.isinf(max_range) or max_range == 0:
            logging.warning("检测到无效的 max_range 值，将 NaN/Inf 或 0 替换为一个小的正值。")
            max_range = 1.0  # 赋予一个默认值以避免错误

        # 现在安全地设置坐标轴限制
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])

    # Plot predictions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_mesh(ax1, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Top View')
    ax1.view_init(elev=90, azim=-90)  # 设置为顶视图
    set_axes_limits(ax1, pred_center, pred_max_range)
    ax1.set_box_aspect([1, 1, 1])

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_mesh(ax2, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction -  Side View')
    ax2.view_init(elev=0, azim=-90)  # 设置为侧视图
    set_axes_limits(ax2, pred_center, pred_max_range)
    ax2.set_box_aspect([1, 1, 1])

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_mesh(ax3, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Front View')
    ax3.view_init(elev=0, azim=0)  # 设置为前视图
    set_axes_limits(ax3, pred_center, pred_max_range)
    ax3.set_box_aspect([1, 1, 1])

    # Plot ground truth
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_mesh(ax4, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Top View')
    ax4.view_init(elev=90, azim=-90)  # 设置为顶视图
    set_axes_limits(ax4, gt_center, gt_max_range)
    ax4.set_box_aspect([1, 1, 1])

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    plot_mesh(ax5, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth -  Side View')
    ax5.view_init(elev=0, azim=-90)  # 设置为侧视图
    set_axes_limits(ax5, gt_center, gt_max_range)
    ax5.set_box_aspect([1, 1, 1])

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_mesh(ax6, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Front View')
    ax6.view_init(elev=0, azim=0)  # 设置为前视图
    set_axes_limits(ax6, gt_center, gt_max_range)
    ax6.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

