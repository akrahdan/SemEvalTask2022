import clozetask.management.tokenize_and_cache as tokenize_and_cache
import clozetask.management.export_model as export_model
import clozetask.management.runscript as runscript
import clozetask.shared.caching as caching
from pathlib import Path
import clozetask.utils.io as py_io

import clozetask.utils.display as display
import os


# export_model.export_model(
#     hf_pretrained_model_name_or_path="roberta-base",
#     output_base_path="./models/roberta-base",
# )
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

conf = tokenize_and_cache.RunConfiguration(
    task_config_path= configs,
    hf_pretrained_model_name_or_path="roberta-base",
    output_dir=f"./cache/{task_name}",
    phases=["train", "val"],
)

tokenize_and_cache.main(args=conf)
#path = os.path.abspath(__file__)

