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

**** Please notice the camera calibration here is the general calibration, if the file does not not include calibraition information, using this one instead. If the data file include calibrated information please use the calibration in the data.

**** Please notice the calibrated information or the data itself might not be paried correctly, Run data_lader.py first for paring checking and calibration checking. 

# Description of the files
  Each file contains Main file, data loader, evaluation file. All the files are similar, only a few function has been modified, that will be specified in later sections.
  
## Data loader file 
contains data loader function and data cheking function.   

Data loader file has changed the input data into tensor data file and paring different data files for supervised learning. Here, data loader is paring image data, radar raw data, ground truth, SPMLX parameters and camera calibration informations. 

Run this file first before any further training. There will be a fina10.log generated shown or somthing like that. Log file is collecting the error, recording which data has the problem will paring. Data_loader.py will skip the error file and it corresboned files. For example if image 1 can't be read, radar 1 and parameter 1 will be skipped. 

*** Please note, if run data_loader directly, log will collect first 10 paired files and show the path of mat file and obj file, value of parameter file to check if the data is paired correctly. Then randomly plot 5 images to check if the performance of the calibration is right or not. 

*** please note the rules of the folder name is following the name on maps3 server if using the data file from the link provided above, the rule of file names needs to be changed in load_mat and load_obj functions.
### Functions include
  gender_info: This part provided with gender information

  load_mat_file: This part loaded mat file which is radar raw file by a name rules, if the path or name is changed, please re-write this part

  load_obj_file: This part loaded obj file which is ground truth file by a name rules, if the path or name is changed, please re-write this part

  extract_number: Take the number of the file name, for sorting

  check_and_match_files: Check if the file and number maches

  load_json_data: Load json file which are SMPLX parameters.

  get_all_file_pairs: paring files base on the index after sorting.

  RF3DPoseDataset: The class to pass the data into model.

  ToTensor: convert data type to tensor

  test_data_loader: To print some sample in log for checking if the paring are correct

  *** main function for data_loader.py
  if __name__ == "__main__":
    root_dir ="DataUsing1"
    file_pairs = get_all_file_pairs(root_dir)
    dataset = RF3DPoseDataset(file_pairs, transform=ToTensor())
    test_data_loader(dataset)
  ***
  change the root_dir for root of the data.
## Main file 
  contains infrmation about model and training parameters.  
  
## Evaluation file 
  contains testing function to calculate metrics. 

