# Refine to the Essence: Less-redundant Skill Learning via Diversity Clustering

RE-Skill integrates the concepts of cluster analysis and policy distillation, clustering similar skills together based on their unique features, learning the most optimal performance within each cluster, and filtering out similar skills that involve excessive and intricate actions, thereby reducing redundancy among skills.
## Dwonstream tasks environments
Before experimenting with the downstream tasks, you need to replace the original gym environment with the gym environment with obstacles we provide.
Note that if you find that HalfCheetah's experiments cause problems with mismatched state dimensions, change
```bash
  exclude_current_positions_from_observation = False
  ```
  field in the HalfCheetah environment file, thus changing HalfCheetah's state dimension to 18.

## Installation
```bash
pip3 install -r requirements.txt
```

## Pretain for primitive skills
```shell
python3 main.py --mem_size=1000000 --env_name="Hopper-v3" --interval=100 --do_train --n_skills=50
```
## Skill clustering
First, create a folder named 'features'
```shell
python skill_cluster.py
```
## Skill distill
First, create a folder named 'distilled_skills_model'
Copy the results of the skill clustering to the cluster value of skill_distill.py file 
```shell
python skill_cluster.py
```
## Downstream task
Copy the results of the skill clustering to the cluster value of run_task.py file 
Then you adjust the env, env_name, states_dim, n_skills in the default configuration appropriately according to your needs
```shell
python run_task.py
```