from preprocess.process import process
from runner.evaluate import evaluate
from runner.predict import predict
from runner.train import train
from web.app import service

import argparse


def main():
    paser = argparse.ArgumentParser(description='商品标题预测')
    paser.add_argument('action',
                       choices=['process', 'train', 'predict', 'evaluate', 'service'],
                       help='操作类型：process | train | evaluate | predict | service')

    args = paser.parse_args()

    if args.action == 'process':
        process()

    elif args.action == 'train':
        train()

    elif args.action == "evaluate":
        evaluate()

    elif args.action == "predict":
        predict()

    elif args.action == "service":
        service()

    else:
        print("未知操作类型，请选择：process / train / evaluate / predict / service")


if __name__ == '__main__':
    main()
