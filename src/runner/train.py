from configuration import config
from preprocess.dataset import get_dataset

from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import torch
from pathlib import Path
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 16
    save_dir: str = './models'
    log_dir: str = "./logs"
    save_steps: int = 100
    early_stop_patience: int = 3
    early_stop_metric: str = 'loss'
    use_amp: bool = True


class Trainer:
    """训练类"""

    def __init__(self, device,
                 model,
                 valid_dataset,
                 collate_fn,
                 compute_metric,
                 train_dataset=None,
                 training_config=TrainingConfig()):
        # 设备
        self.device = device
        # 模型
        self.model = model.to(device)
        # 数据
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # 数据整理器
        self.collate_fn = collate_fn
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
        # 计算评估指标
        self.compute_metric = compute_metric
        # 训练参数
        self.training_config = training_config

        # 全局step(训练的batch数)
        self.step = 0

        # tensorboard
        self.writer = SummaryWriter(str(Path(training_config.log_dir) / time.strftime('%Y-%m-%d_%H-%M-%S')))

        # 初始化best_score
        self.best_score = -float('inf')
        # 初始化早停计数
        self.early_stop_count = 0

        # 缩放器
        self.scaler = torch.amp.GradScaler(device='cuda', enabled=training_config.use_amp)

    def _get_dataloader(self, dataset):
        dataset.set_format(type='torch')
        generator = torch.Generator()
        generator.manual_seed(42)
        return DataLoader(dataset,
                          batch_size=self.training_config.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          generator=generator)

    def _train_one_step(self, inputs):
        self.model.train()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.amp.autocast(device_type=self.device.type,
                                dtype=torch.float16,
                                enabled=self.training_config.use_amp):
            outputs = self.model(**inputs)
            loss = outputs.loss  # outputs.logits

        # 反向传播
        self.scaler.scale(loss).backward()
        # 更新参数
        self.scaler.step(self.optimizer)
        # 更新缩放系数
        self.scaler.update()
        # 梯度清零
        self.optimizer.zero_grad()

        return loss.item()

    def _load_checkpoint(self):
        path = Path(self.training_config.save_dir) / 'checkpoint' / 'checkpoint.pt'
        if path.exists():
            tqdm.write('发现检查点，继续训练')
            checkpoint = torch.load(path)

            self.model.load_state_dict(checkpoint['model'])
            self.step = checkpoint['step']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.best_score = checkpoint['best_score']
            self.early_stop_count = checkpoint['counter']
        else:
            tqdm.write('没有检查点，重新开始训练')

    def _save_checkpoint(self):
        """
        保存检查点

        1、模型权重
        2、step
        3、优化器状态
        4、缩放因子
        5、最优分数
        6、早停计数
        """
        checkpoint = {
            "model": self.model.state_dict(),
            'step': self.step,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_score': self.best_score,
            'counter': self.early_stop_count
        }
        torch.save(checkpoint, Path(self.training_config.save_dir) / 'checkpoint' / 'checkpoint.pt')

    def _should_save_or_stop(self, metrics) -> bool:
        metric = metrics[self.training_config.early_stop_metric]
        score = -metric if self.training_config.early_stop_metric == 'loss' else metric
        # 判断是否保存
        if score > self.best_score:
            self.early_stop_count = 0
            self.best_score = score
            tqdm.write("保存模型")
            self.model.save_pretrained(self.training_config.save_dir)
            return False
        # 判断是否早停
        else:
            self.early_stop_count += 1
            if self.early_stop_count <= self.training_config.early_stop_patience:
                return False
            else:
                return True

    def evaluate(self) -> dict:
        """验证评估"""
        self.model.eval()

        total_loss = 0
        all_predictions = []
        all_labels = []

        dataloader = self._get_dataloader(self.valid_dataset)
        for inputs in tqdm(dataloader, desc='Evaluation'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                # 前向传播
                outputs = self.model(**inputs)

            loss = outputs.loss
            total_loss += loss.item()

            # 预测值
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.tolist())

            # 真实值
            labels = inputs['labels']
            all_labels.extend(labels.tolist())

        # 验证集平均loss
        loss = total_loss / len(dataloader)
        metrics = self.compute_metric(all_predictions, all_labels)

        return {'loss': loss, **metrics}

    def train(self):
        # 加载检查点
        self._load_checkpoint()

        dataloader = self._get_dataloader(self.train_dataset)
        current_step = 0
        for epoch in range(1, 1 + self.training_config.epochs):
            for inputs in tqdm(dataloader, desc=f'Epoch:{epoch}'):
                current_step += 1
                if current_step <= self.step:
                    continue
                self.step += 1
                loss = self._train_one_step(inputs)
                if self.step % self.training_config.save_steps == 0:
                    tqdm.write(f'Epoch;{epoch},Loss:{loss},Step:{self.step}')
                    self.writer.add_scalar('LOSS', loss, self.step)

                    # 拿到验证集评估指标并打印
                    metrics = self.evaluate()
                    metrics_str = ' | '.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'Evaluation: {metrics_str}')

                    # 根据评估指标判断保存模型和早停
                    if self._should_save_or_stop(metrics):
                        tqdm.write('早停')
                        return

                    # 保存检查点
                    self._save_checkpoint()


def train():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取all_labels
    with open(config.SAVE_MODELS_DIR / 'labels.txt', 'r', encoding='utf-8') as f:
        all_labels = f.read().splitlines()

    id2label = {index: label for index, label in enumerate(all_labels)}
    label2id = {label: index for index, label in enumerate(all_labels)}

    # 模型
    model = BertForSequenceClassification.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese',
                                                          num_labels=len(all_labels),
                                                          id2label=id2label,
                                                          label2id=label2id)

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

    # 数据
    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('valid')

    # 数据整理器
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

    # 定义评估指标
    def compute_metric(predictions, labels) -> dict:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {'accuracy': accuracy, 'f1': f1}

    # 超参数
    training_config = TrainingConfig(save_dir=config.SAVE_MODELS_DIR, log_dir=config.LOGS_DIR)

    trainer = Trainer(device=device,
                      model=model,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      collate_fn=collate_fn,
                      compute_metric=compute_metric,
                      training_config=training_config)

    trainer.train()


if __name__ == '__main__':
    train()
