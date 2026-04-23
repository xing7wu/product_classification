from configuration import config

from datasets import load_dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer


def process():
    """数据预处理"""

    # 加载数据
    dataset_dict = load_dataset(
        'csv',
        data_files={
            'train': str(config.RAW_DATA_DIR / 'train.txt'),
            'test': str(config.RAW_DATA_DIR / 'test.txt'),
            'valid': str(config.RAW_DATA_DIR / 'valid.txt'),
        },
        features=Features({'label': Value('string'), 'text_a': Value('string')}),
        delimiter='\t')

    # 数据清洗
    dataset_dict = dataset_dict.filter(lambda x: x['label'] is not None and x['text_a'] is not None)

    # 处理label（标签）
    all_labels = sorted(set(dataset_dict['train']['label']))
    dataset_dict = dataset_dict.cast_column('label', ClassLabel(names=all_labels))

    # 保存all_labels，定义模型时会用到
    with open(config.SAVE_MODELS_DIR / 'labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_labels))

    # 处理text_a（商品标题）
    ## 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

    def tokenize(batch):
        """分词函数"""
        inputs = tokenizer(batch['text_a'], truncation=True, return_token_type_ids=False)
        inputs['labels'] = batch['label']
        return inputs

    dataset_dict = dataset_dict.map(tokenize, remove_columns=['label', 'text_a'])

    # 保存预处理数据
    dataset_dict.save_to_disk(str(config.PROCESSED_DATA_DIR))


if __name__ == '__main__':
    process()
