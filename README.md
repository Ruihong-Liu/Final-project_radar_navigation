# Final-project_radar_navigation
Describtion: This paper presents the SimpleCFA model, a neural network that estimates 3D human meshes under challenging conditions such as poor lighting and low visibility environments, including fog and smoke. Millimeter-wave (mm-wave) radar signals can
penetrate clothing and reflect off the human body. Also, mm-Wave has the benefit of
detecting the small movement of the human body under high-frequency conditions.This
model introduces a method for predicting human meshes using only radar signals as
input data. Furthermore, the model is designed to work with a single person

Three files are uploaded. Those are the three versions of the project. In these files, Final11_multi_head_attention and Final11_single_head_attention are the one that not cameracalibrated. Final11_full_camera calibrated is using the vertices after camera calbration. But Final11_full_camera calibrated are still on going, has the problem with ploting meshes

# Data collection and informations.
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

**** Please notice the camera calibration here is the general calibration, if the file does not not include calibraition information, using this one instead. If the data file include clibrated information please use the calibration in the data.

# Description of the files
  Each file contains Main file, data loader, evaluation file. All the files are similar, only a few function has been modified, that will be specified in later sections.
## Data loader file 
contains data loader function and data cheking function.   Data loader file has changed the input data into tensor data file and paring different data file for supervised learning. 

## Main file 
  contains infrmation about model and training parameters.  
## Evaluation file 
  contains testing function to calculate metrics. 

