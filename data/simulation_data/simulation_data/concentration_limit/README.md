# Simulation data for the simulation result presented in Section 3.1 in the article

***All the data in the five folders is not included now, but it can be obtained from the authors upon reasonable request. You can generate the all the data by yourself using the **generate_simulation.mlapp** in the **code** folder.***

Here is part of the source text in Section 3.1:

> We model a 3D particle field, with 60 particles with diameters 20-30 μm, 63 layers with  an axial layer interval of 30 μm, and an axial distance range of 1800-3660 μm. We take 26.82  μm as the lateral distance of a voxel. It is worth pointing out that, in the simulation, the  transformed reconstructed images are obtained by Holo-processing with the original  parameters (0, 0, 1), given the fact that there is no experimental data involved here. Because  each hologram contains 256×256 pixels with a pixel pitch of 2.2 μm, each group of data thus  corresponds to a particle field of 563×563×1860 μm3 (21×21×63 voxels). 128 groups of  training data are prepared, corresponding to data shape of 256×256×8064, and the 100  epochs of training take 135 minutes with a learning rate of 1×10-4.  
>
> In the test, we prepare 9 groups of data and spliced them in a horizontal 3×3 way, so the  particle field size after splicing is 63×63×63 voxels. Since each group of data contains 60  particles, this particle field contains 540 particles, that is to say, the ppp concentration is about  1.36×10-1.

---

The training input data is inside the **train_in** folder and the corresponding training label is inside the **train_out** folder.

## train_in

There are totally 8064 images with the size of 256×256 inside. Each image is named according to their property. For example, "re1_60Num_63Slice_at1800um.bmp" means that this image belongs to group 1, inside which are 60 particles distributed in 63 axial slice or layer, and this image's axial reconstruction distance is 1800 μm. The 8064 images consist of 128 groups, with 63 reconstruction images in each group.

## train_out

The images inside this folder are one-to-one to the images in the train_in folder. For instance, "pr1_60Num_63Slice_at1800um.bmp" corresponds to "re1_60Num_63Slice_at1800um.bmp" mentioned above.

---

The test input data is inside the **test_in** folder and the corresponding prediction result by the U-Net is put in the **test_out** folder. The ground truth for the test is inside the **test_truth** folder.

The trained model for testing is **concentration_limit_model.h5**. <u>It is not included now, but it can be obtained from the authors upon reasonable request.</u> 

Here is the corresponding test result presented in the article:

![极限浓度成图增强版](../imgs/极限浓度成图增强版.bmp)







