# Subjective Attributes
This repository contains the code for the MM'19 paper "Learning Subjective Attributes of Images from Auxiliary Sources".
The code allowing training a model to learn subjective attributes on the marketing and personality datasets.

## Running the code
1. download and unzip the folder data in the main folder
2. cd to the main folder 
3. run the following command:
```
python train.py --attr_path ../data/attributes/<attribute list file> --dataset_path ../data/<dataset folder>/
```
where:
- attribute list file is the .txt file specify a sublist of subjective attributes to be learned simultaneously during training. The sublist of attributes can include any of the headers in the aux_data.csv
- dataset folder is either brand_dataset or personality dataset

### Other training options
- gpu: specifies gpu id
- fixed_std: if using fixed_std (equals by default to 0.1) against a learned std
- resume_training: file to which resume training from

### Training hyperparameters
These are indicated in the file data/settings.json and include number of epochs, learning rate, batch_size and minimum number of images to filter brands/users

### Required dependencies to run the code:
- [Pytorch](https://pytorch.org/)
- [tensorboard_logger](https://pypi.org/project/tensorboard_logger/)
- [tqdm](https://github.com/tqdm/tqdm)
