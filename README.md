# LBS： Learning from Better Simulation

Authors: Bozhen Zhou (zbz22@mails.tsinghua.edu.cn) and Ping Su (su.ping@sz.tsinghua.edu.cn)

*Tsinghua Shenzhen International Graduate School, Tsinghua University*

---

***All the codes are in the code folder. Note that the data in the data folder here is just a part for exhibition. All the experimental data and some of the simulation data are in the master branch.  You can generate all the simulation data by yourself by running the generate_simulation.mlapp in the code folder. If you still need additional data, feel free to reach out to the authors.***


## Abstract

Deep learning has been widely used in the field of particle holography because it can greatly improve reconstruction efficiency. Obtaining the ground truth of a real particle field is almost impossible, which makes the results by traditional deep learning methods unreliable, especially for particle fields with high concentration. However, the multiple-scattering process by the dense particles is too complex to be modeled either physically or by a neural network. To address this issue, we present a **learning from better simulation (LBS) method** in this paper. Utilizing the physical information from experimentally captured hologram by an optimization method, the LBS method bypasses the multiple-scattering process and creates highly realistic synthetic data for training a simple U-Net. Training the model entirely on synthetic data, the U-Net can quickly generalize to experimental data without any manual labeling. We present simulation and experimental results to demonstrate the effectiveness and robustness of the proposed technique in the highest-concentration scenario by far. The proposed method paves the way toward reliable particle field imaging in high concentration and can potentially guide the design of optimization toward simulation data in other fields.

## Requirements

These are the recommended versions:

> The U-Net architecture is implemented using Keras 2.6.0 with the TensorFlow-GPU 2.6.0 backend. Other software environments are Python 3.7, CUDA 11.2, CUDNN 8.1, and MATLAB R2022b. The network training is conducted on a Nvidia RTX 3090 24G GPU and the generation of training data is conducted on an Intel Core i5-11400H CPU.

## Quick Start
Please click *download ZIP* button in the main branch and in the master branch respectively. Note that the **experiment_data.rar** in the master branch may be too big, so sometimes after clicking the *down load ZIP* button, you may not get a correct rar file. If that happens, we suggest that you get to the **experiment_data.rar** file's website and click the raw button:
![warning](https://github.com/zhoubozhen/LBS/blob/main/imgs/warning.jpg)
1. Get all the data in the master branch first and unzip them all. Put all the data into the data folder. Replace all the data originally inside with the full version you get from the master branch.

2. After that, inside your data folder are two folders, namely the *experiment_data* and *simulation_data* folder.
3. In the code folder, run the programs according to the README there.
