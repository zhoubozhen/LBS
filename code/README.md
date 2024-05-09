# Code for implementation

All the main programs are here. All the utilities functions are inside the **function** folder for callbacks.

## generate_simulation.mlapp

This is a MATLAB APP for the simulation data generation for the U-Net training and testing.

![Generation_app](F:\全息工作\英文论文\数据与代码\imgs\Generation_app.bmp)

The three parameters **N_R, N_sigma and Gamma factor** are the found optimal parameters in the OCE method. It is worth pointing out that the N_R is inversed. For example, the optimal N_R is -0.3068 while the value should be 0.3068 here.

The **Group number** is the amount of groups included in your data. Each group is corresponding to one hologram. The **Axial interval, Start distance, Minimum diameter, Maximum diameter and End distance** are in μm. The **Particle number** denotes the number of particles included in one hologram. 

The **Group index** is the serial number of the group data you want to create. For example, you want to create one group of data for testing, and you call this group as "group 1". Set the number as 1 and click **Create folder** button and you will find one new folder in the path "../data/simulation_data/".

Click the **Run** button to run this program according to the set parameters. After that, you will find newly generated data in the folder "data/generation_data". If that is what you want, click the **Data migration** button to copy the data to the new folder you have just created. In the end, you can click the **Delete** button to delete all the data in the pre-created folder "generation_data".

## compound_cri.m

This is a main function for the compound focusing criterion.

## compare_exp.mlapp

This is a MATLAB APP for the comparison between the prediction result and the reference result in the experiment, corresponding to Section 3.2 in the article.

![compare_exp](F:\全息工作\英文论文\数据与代码\imgs\compare_exp.bmp)

Click the **Run** button and you will get the result like this:

![compare_exp_result](F:\全息工作\英文论文\数据与代码\imgs\compare_exp_result.bmp)

The positive is the TP, the negative is the FP and the omitted is the FN.

## compare_sim.mlapp

This is a MATLAB APP for the test in the simulation result corresponding to Section 3.1 in the article.

![compare_sim](F:\全息工作\英文论文\数据与代码\imgs\compare_sim.bmp)

Click **Run** button to run the program and get a result like this:

![compare_sim_result](F:\全息工作\英文论文\数据与代码\imgs\compare_sim_result.bmp)

In MATLAB Workspace you will get:

![sim_result_sample](F:\全息工作\英文论文\数据与代码\imgs\sim_result_sample.bmp)

The two pack are the ground truth pack and the prediction result pack, respectively. For example, each row corresponds one particle and the five columns are representative of lateral position, axial position, diameter and the group series number.

## delete_particle.m

Run the **compare_exp.mlapp**  first to get the data and run this program to delete additional particles overlapping in the axial direction.

## Optimization.m

This is the Optimization module in the OCE method.

## train_Unet.py

This is the main program to train the modified U-Net. Follow the tips in the beginning to make some adjustments.

Here is the recommended versions:

> The U-Net architecture is implemented using Keras 2.6.0 with the TensorFlow-GPU  2.6.0 backend. Other software environments are Python 3.7, CUDA 11.2, CUDNN 8.1, and  MATLAB R2022b. The network training is conducted on a Nvidia RTX 3090 24G GPU and  the generation of training data is conducted on an Intel Core i5-11400H CPU.

## test_Unet.py

This is for testing the trained U-Net model. Follow the tips inside.
