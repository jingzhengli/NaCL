## NaCL:Noise-Robust Cross-Domain Contrastive Learning for Unsupervised Domain Adaptation
Published in Machine Learning 2023.
### Introduction
This is a PyTorch implementation of ["NaCL:Noise-Robust Cross-Domain Contrastive Learning for Unsupervised Domain Adaptation"]. 

### Requirements
* Python 3.7
* torchvision 0.9.0  
* PyTorch 1.8.0


### Train:

- Unsupervised DA on `Office31`,`OfficeHome`, and `VisDA2017` datasets:
    ```bash
    sh runUDA.sh
    ```
- Unsupervised DA on `ImageNet-scale` dataset:
    ```
    python main.py --dataset_root ./data/ --src IN --tgt INR --contrast_dim 256 --module domain_loss --cw 1 --lr 0.003 --batch_size 32 --max_key_size 20 --max_iterations 50000
    ```
- Semi-supervised DA on `COVID-19` dataset:
    ```bash
    sh runSSDA.sh
    ```
### Log:

- The training log will be generated in the folder with ``--log_dir``. We can visualize the training process through `tensorboard` as follows.

    ```bash
    tensorboard --logdir=/log_dir/ --host= `host address`
    ```

### Usage

- We uploaded the file `PythonGraphPers_withCompInfo.so` for computing the `connected components`. If you need to generate it, you can compile the C++ code in folder `ref`, run `./compile_pers_lib.sh` (by default it requires Python 3.7. If you are using other Python versions, modify the command inside `compile_pers_lib.sh`).

- `pybind11` is a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code. Our code will be further improved to make it cleaner and easier to use.


***Note***: Place the datasets in the corresponding data path.



