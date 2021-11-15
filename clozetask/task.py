from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from clozetask.lib.feature_template import single_sentence_featurize, labels_to_bimap
from clozetask.utils.io import read_csv
from .core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    SuperGlueMixin,
    Task,
    TaskTypes
)

@dataclass
class Example(BaseExample):
    guid: str
    passage_text: str
    label: str
    
    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid= self.guid,
            passage_text=tokenizer.tokenize(self.passage_text),
            label_id= SemevalTask.LABEL_TO_ID[self.label]
            
        )

@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    passage_text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.passage_text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow
        )

@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list

@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class SemevalTask(SuperGlueMixin, Task):
    Example = Example
    
    TokenizedExample = Example
    val_df = None
    DataRow = DataRow
    Batch = Batch
    Task_Type = TaskTypes.CLASSIFICATION
    LABELS = ["0", "1", "2"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        
        train, val =read_csv(self.train_path, self.label_path)
        self.val_df = val
        return self._create_examples(lines=train, set_type="train")

    def get_val_examples(self):
        if self.val_df:
            return self._create_examples(lines=self.val_df, set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_csv(self.test_path, self.label_path), set_type="test")
    
    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(
                Example(
                    # NOTE: get_glue_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, i),
                    passage_text=line[0],
                    label=str(line[1]) if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
    
    @staticmethod
    def super_glue_format_preds(pred_dict):
        return pred_dict["preds"]

