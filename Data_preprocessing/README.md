# Wojood Pre-processing
In the context of the Arabic NER SharedTask 2023, this repository contains all the pre-processing scripts on the Wojood dataset.  
In order to use the Wojood dataset with the models, the OIB-formatted manifest must be converted to a compatible format. 
`PIQN` and `DiffusionNER` share the same JSON manifest format requirement.  
This folder also contains the data re-sampling scripts used.

## How to run:
### Requirements:
To run the scripts and notebooks in this repository, create an environment by running the following commands:
```shell
conda create -n arabicNER-preprocessing python=3.10
conda activate arabicNER-preprocessing
pip install requirements.txt
```
### Re-sampling:
Simply follow the steps in `NER_Adaptive_Resampling/resampling.ipynb`.

### PIQN and DiffusionNER:
[PIQN paper](https://arxiv.org/abs/2203.10545)  
[DiffusionNER paper](https://arxiv.org/abs/2305.13298)  
Make sure the conda environment just created is activated. Then run following commands:
```shell
cd wojood_preprocess_for_piqn

# If you want to convert all files in a specified folder:
python preprocess_wojood.py --dataset-directory <folder_path> --save-directory ./save

# If you want to convert a single file:
python preprocess_wojood.py --file <file_path> --save-directory ./save
```