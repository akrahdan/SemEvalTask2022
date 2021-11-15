import clozetask.management.tokenize_and_cache as tokenize_and_cache
import clozetask.management.export_model as export_model
import clozetask.management.runscript as runscript
import clozetask.shared.caching as caching

import clozetask.utils.io as py_io

import clozetask.utils.display as display
import os

configs = {
  "task": "classTask",
  "paths": {
    "train": "/content/data/data/Clarification.tsv",
    "label": "/content/data/label.tsv",
   
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