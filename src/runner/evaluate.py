from configuration import config
from preprocess.dataset import get_dataset
from runner.train import TrainingConfig, Trainer

import torch
from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score


def evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型
    model = BertForSequenceClassification.from_pretrained(config.SAVE_MODELS_DIR)

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

    # 数据
    test_dataset = get_dataset('test')

    # 数据整理器
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

    # 评估指标
    def compute_metric(predictions, labels) -> dict:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {'accuracy': accuracy, 'f1': f1}

    # 超参数
    training_config = TrainingConfig(save_dir=config.SAVE_MODELS_DIR, log_dir=config.LOGS_DIR)

    trainer = Trainer(device=device,
                      model=model,
                      valid_dataset=test_dataset,
                      compute_metric=compute_metric,
                      collate_fn=collate_fn,
                      training_config=training_config
                      )
    metrics = trainer.evaluate()

    return metrics


if __name__ == '__main__':
    print(evaluate())
