import torch


class Predictor:
    def __init__(self, device, model, tokenizer):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer

    def predict(self, text: str | list):
        """预测"""
        is_str = isinstance(text, str)
        if is_str:
            text = [text]

        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 模型输出 → 分类标签
        logits = outputs.logits  # is a tensor
        predictions = torch.argmax(logits, dim=-1)
        res = [self.model.config.id2label[prediction] for prediction in predictions.tolist()]

        if is_str:
            return res[0]
        else:
            return res
