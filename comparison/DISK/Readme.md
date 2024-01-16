## Requirements
We assume you have access to a gpu that can run CUDA 11.3. Then, the simplest way to install all required dependencies is to create an anaconda environment and activate it:
```shell
conda env create -f conda_env.yml
conda activate disk
```

## Train skills
Note the change to the model_path path in the train.py file
```shell
python train.py env=hopper
```

## Downstream tasks
Note the changes to the Hopper and HalfCheetah initialization in the run_task file
```shell
python run_task.py
```