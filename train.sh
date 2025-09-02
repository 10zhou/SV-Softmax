#!/usr/bin/env bash

#python train_demo.py --arch resnet18 --method ce --epochs 200 --batch-size 128 --data-root ../datasets
#python train_demo.py --arch resnet18 --method arcface --epochs 200 --batch-size 128 --data-root ../datasets
#python train_demo.py --arch resnet18 --method normface --epochs 200 --batch-size 128 --data-root ../datasets
#python train_demo.py --arch resnet18 --method virtual_softmax --epochs 200 --batch-size 128 --data-root ../datasets
python train_demo.py --arch resnet18 --method svsoftmax --epochs 200 --batch-size 128 --data-root ../datasets
