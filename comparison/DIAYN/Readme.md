## Pretain for primitive skills
```shell
python3 main.py --mem_size=1000000 --env_name="Hopper-v3" --interval=100 --do_train --n_skills=50
```

## Downstream task
Note that the trained skills are adjusted to the Checkpoint corresponding env directory
```shell
python run_task.py
```