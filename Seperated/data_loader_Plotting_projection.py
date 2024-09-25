import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch
import random
# set log information path
log_file_path = 'data11_log_cali.txt'

if log_file_path:
    directory = os.path.dirname(log_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path)])
logger = logging.getLogger('data_loader')
logger.setLevel(logging.INFO)

# project the 3D points to 2D image
def project_to_2d(points_3d, camera_matrix, camera_extrinsics):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_camera = np.dot(camera_extrinsics, points_3d_homogeneous.T).T
    points_image_homogeneous = np.dot(camera_matrix, points_camera[:, :3].T).T
    points_2d = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2, np.newaxis]
    return points_2d

#plot the 3D points on 2D images
def plot_projected_image(image, projected_vertices, sample_idx):
    try:
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            image = image.transpose(1, 2, 0)  # CHW to HWC

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
        plt.title(f'Projected Vertices on Image {sample_idx}')
        plt.axis('off')
        plt.savefig(f"projected_image_{sample_idx}.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error during plotting projected image for sample {sample_idx}: {e}")

# randomly select 5 images to plot
def plot_random_samples(dataset, num_samples=5):
    try:
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            sample = dataset[idx]
            if sample is not None:
                plot_projected_image(sample['image'], sample['projected_vertices'], idx)
    except Exception as e:
        logger.error(f"Error during random sample plotting: {e}")
