import logging
import os
from importlib import import_module
from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer
)
from train.utils_ner import Split, TokenClassificationDataset, TokenClassificationTask

logger = logging.getLogger(__name__)


# thông số cho chạy script
# --------------------------
# vị trí file validation/test
data_dir = 'H:/Work from home/better-kw-15455-data'
# task NER của Transformers
task_type = 'NER'
# vị trí model bert
model_name_or_path = 'H:/Work from home/real_estate_ner_kw_15455'
# vị trí file labels (O, CDT, DA)
labels_path = 'H:/Work from home/labels.txt'
# vị trí lưu kết quả prediction (f1_score và predictions)
output_dir = 'H:/Work from home'


def main():

    # -----------------------------------------------
    # gọi token_classfication_task
    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )
    # -----------------------------------------------

    # -----------------------------------------------
    # tạo label cho prediction
    labels = token_classification_task.get_labels(labels_path)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    # -----------------------------------------------

    # -----------------------------------------------
    # config cho models (tạo tokenizer, labels, ...)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    # -----------------------------------------------

    # -----------------------------------------------
    # trả ra predictions labels theo dạng list tương ứng với word
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list
    # -----------------------------------------------

    # -----------------------------------------------
    # tính f1-score
    # TODO: thêm f1 riêng từng tag
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
    # -----------------------------------------------

    # Initialize our Trainer
    # hàm trainer của transformers, cơ bản giúp cho việc xử lý model (mới được hugging face thêm vào)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    # ------------------------------------------------------------

    # -----------------------------------------------
    # khởi tạo dữ liệu cho predictions
    test_dataset = TokenClassificationDataset(
        token_classification_task=token_classification_task,
        data_dir=data_dir,  # location to folder with file name test.txt
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        mode=Split.test,
    )

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    preds_list, _ = align_predictions(predictions, label_ids)
    # -----------------------------------------------

    # Save f1_score
    output_test_results_file = os.path.join(output_dir, "test_results.txt")
    if trainer.is_world_process_zero():
        with open(output_test_results_file, "w") as writer:
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Save predictions
    output_test_predictions_file = os.path.join(output_dir, "test_predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_dir, "test.txt"), "r") as f:
                token_classification_task.write_predictions_to_file(writer, f, preds_list)


if __name__ == "__main__":
    main()
