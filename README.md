# Final-project_radar_navigation
Describtion: This paper presents the SimpleCFA model, a neural network that estimates 3D human meshes under challenging conditions such as poor lighting and low visibility environments, including fog and smoke. Millimeter-wave (mm-wave) radar signals can penetrate clothing and reflect off the human body. Also, mm-Wave has the benefit of detecting the small movement of the human body under high-frequency conditions.This model introduces a method for predicting human meshes using only radar signals as input data. Furthermore, the model is designed to work with a single person Three files are uploaded. Those are the three versions of the project. 

In these files, Final11_multi_head_attention and Final11_single_head_attention are the one that not cameracalibrated. Final11_full_camera calibrated is using the vertices after camera calbration. But Final11_full_camera calibrated are still on going, has the problem with ploting meshes

# Data collection and informations.
On the server, Datas are under the path /mnt/data-B/Ruihong_radar. Within this path. 

DataFinal is all the data donwloaded from BaiDU cloud, DataSet is original ZiP file, DataUsing is 7 peoples data can directly used with data_loader rules. DataUsing1 is one complete people data for testing, DataUsing2 are two conplete people data. DataUsing3 are one action for each two people, which is the smallest sample to test.

The unprocessed Radar & RGB-D data is available at: 

link: https://pan.baidu.com/s/1muGqz3sHmNDJU_CWt2aHvA 

code: f5g8

Vicon data is available at: 

link:https://pan.baidu.com/s/1euV5gBUYhxw2kgCDZGINGQ 

code: ahya

Data: mmWave radar data, including RF signals, XYZ radar tensor (heatmap), point cloud, and RGB-D camera images

Annotation: human body mesh, human body keypoints

Size: 500k in total from 20 participants performing 50 activities

Calibration:
gb_matrix=np.array([[375.66860062,   0.      ,   319.99508973], [  0.    ,     375.66347079 ,239.41364796], [  0.     ,      0.      ,     1.        ]])
radar2rgb_tvec=np.array([-0.03981857,1.35834002,-0.05225502])
radar2rgb_rotmatrix=np.array([[ 9.99458797e-01,  3.28646073e-02,  1.42475954e-03], [4.78233954e-04,  2.87906567e-02, -9.99585349e-01], [-3.28919997e-02,  9.99045052e-01,  2.87593582e-02]])

gb_matrix denotes the intrinsic matrix of the RGB-D camera, radar2rgb_tvec and radar2rgb_rotmatrix indicate the coordinates transformation matrix from radar to camera.

**** Please notice the camera calibration here is the general calibration, if the file does not not include calibraition information, using this one instead. If the data file include calibrated information please use the calibration in the data.

**** Please notice the calibrated information or the data itself might not be paried correctly, Run data_lader.py first for paring checking and calibration checking. 

# Description of the files
  Each file contains Main file, data loader, evaluation file. All the files are similar, only a few function has been modified, that will be specified in later sections.

***

Step1: Run data_loader_loader_main to test if the dataset is valid and check the sample data if paried correctly

Step2: Run Train_and_model_main_train for training and validate the model

step3: Find the best model which has the smallest loss and using evaluation function for testing.

*** 

## data_loader_Load_data
  Contains all the loading functions. For other dataset, the rules of file names may change, therefore, this function may needs to be rewrite. 
  
## data_loader_camera_calibration
  contains camera calibration informations and calculations
  
## data_loader_Plotting_projection
  contains the projection function and plotting function
  
## data_loader_loader_main
  the main file for the data preperation section, transfer data into tensor and paring for model training.

## Train_and_model_model
  Contains the setting up of the models including FPN, Multi-attention and SMPL-X
  
## Train_and_model_loss
  The calculation of loss function of the model
  
## Train_and_model_plotting_3D_mesh
  The function about how to plot the 3D mesh
  
## Train_and_model_main_train
  THe function including training, validation, testing and ooptimizeing.

## evaluation
  Metics for evalating the performance of the model.
