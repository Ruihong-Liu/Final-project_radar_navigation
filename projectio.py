import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def read_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
    return np.array(vertices)

def main():
    # 文件路径
    cam_ins_path = 'DataUsing1/P1/ins_ext/cam_ins.txt'
    cam_ext_path = 'DataUsing1/P1/ins_ext/vicon_to_cam.txt'
    radar_rot_path = 'DataUsing1/P1/ins_ext/radar2rgb_rotmatrix.txt'
    radar_tvec_path = 'DataUsing1/P1/ins_ext/radar2rgb_tvec.txt'
    image_path = 'DataUsing1/P1/actions/2/rgb/41.png'
    obj_file_path = 'Saving_versions/Version3_CNN_FPN_Resnet_Attention/results/testing/prediction_test_0.obj'  # 假设您的OBJ文件名为model.obj

    # 读取矩阵
    cam_ins = np.loadtxt(cam_ins_path)
    cam_ext = np.loadtxt(cam_ext_path)
    radar_rot = np.loadtxt(radar_rot_path).reshape(3, 3)
    radar_tvec = np.loadtxt(radar_tvec_path).reshape(3, 1)
    
    # 读取OBJ文件中的顶点数据
    vertices = read_obj(obj_file_path)
    print("Vertices from OBJ:\n", vertices)

    # 将OBJ顶点转换到相机坐标系
    vertices_cam = radar_rot @ vertices.T + radar_tvec
    print("Vertices in camera coordinate system:\n", vertices_cam)

    # 投影到2D图像平面
    uvw = cam_ins @ vertices_cam
    uvw /= uvw[2, :]  # 使用广播机制确保按列归一化
    uvs = uvw[:2, :].T
    print("Projected points in image plane:\n", uvs)

    # 显示图像并绘制投影点
    image = mpimg.imread(image_path)
    print("Image shape:", image.shape)
    
    # 映射到图像坐标系
    uvs[:, 0] = (uvs[:, 0] - np.min(uvs[:, 0])) / (np.max(uvs[:, 0]) - np.min(uvs[:, 0])) * image.shape[1]
    uvs[:, 1] = (uvs[:, 1] - np.min(uvs[:, 1])) / (np.max(uvs[:, 1]) - np.min(uvs[:, 1])) * image.shape[0]

    # 绘制投影点
    plt.imshow(image)
    plt.scatter(uvs[:, 0], image.shape[0] - uvs[:, 1], s=10, c='red', marker='o')
    plt.title('Projected OBJ Vertices on Image')
    plt.show()

if __name__ == '__main__':
    main()