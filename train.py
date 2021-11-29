import clozetask.management.tokenize_and_cache as tokenize_and_cache
import clozetask.management.export_model as export_model
import clozetask.management.runscript as runscript
import clozetask.shared.caching as caching
from pathlib import Path
import clozetask.utils.io as py_io
import clozetask.management.scripts.configurator as configurator
import clozetask.utils.display as display
import os


export_model.export_model(
    hf_pretrained_model_name_or_path="google/electra-base-discriminator",
    output_base_path="./models/electra-base",
)
BASE_DIR = Path(__file__).resolve().parent
configs = {
  "task": "classTask",
  "paths": {
    "train": f"{BASE_DIR}/data/Clarification.tsv",
    "label": f"{BASE_DIR}/data/labels.tsv",
   
  },
  "name": "classTask"
}

task_name = "classTask"
task_path = Path("./tasks/configs").mkdir(parents=True, exist_ok=True)

py_io.write_json(configs, "./tasks/configs/classTask_config.json")

conf = tokenize_and_cache.RunConfiguration(
    task_config_path= configs,
    hf_pretrained_model_name_or_path="google/electra-base-discriminator",
    output_dir=f"./cache/{task_name}",
    phases=["train", "val"],
)

tokenize_and_cache.main(args=conf)


jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path="./tasks/configs",
    task_cache_base_path="./cache",
    train_task_name_list=["classTask"],
    val_task_name_list=["classTask"],
    train_batch_size=8,
    eval_batch_size=16,
    epochs=3,
    num_gpus=1,
).create_config()
os.makedirs("./run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, "./run_configs/classTask_run_config.json")
display.show_json(jiant_run_config)

run_args = runscript.RunConfiguration(
    jiant_task_container_config_path="./run_configs/classTask_run_config.json",
    output_dir="./runs/classTask",
    hf_pretrained_model_name_or_path="google/electra-base-discriminator",
    model_path="./models/electra-base/model/model.p",
    model_config_path="./models/electra-base/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=500,
    do_train=True,
    do_val=True,
    do_save=True,
    force_overwrite=True,
)
runscript.run_loop(run_args)
#path = os.path.abspath(__file__)

