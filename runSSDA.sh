#!/usr/bin/env bash

python main_covid.py --network resnet18 --batch_size 32

python main_covid.py --network resnet50 --batch_size 32

python main_covid.py --network resnet101 --batch_size 32
