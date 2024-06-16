import wandb
import os

# os.environ["WANDB_START_TIMEOUT"] = "300"
wandb.init(project="approxWMC_test", mode='offline', settings=wandb.Settings(_disable_stats=True, _disable_service=True, console='auto'))
wandb.log({"test_metric": 1})
print("!!!")
wandb.finish()

