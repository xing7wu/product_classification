from configuration import config
from runner.predict import Predictor
from web.service import TitleClassify
from web.schemas import Title, Category

from fastapi import FastAPI, Query
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import uvicorn

app = FastAPI(description='商品分类预测')

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型
model = BertForSequenceClassification.from_pretrained(config.SAVE_MODELS_DIR)
# 分词器
tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

predictor = Predictor(device=device,
                      model=model,
                      tokenizer=tokenizer)
# 标题分类器
title_classify = TitleClassify(predictor=predictor)


@app.post('/title_classify')
def predict(title: Title) -> Category:
    res = title_classify.predict(title.name)
    return Category(name=res)


def service():
    uvicorn.run('web.app:app', host='127.0.0.1', port=8001)


if __name__ == '__main__':
    service()
