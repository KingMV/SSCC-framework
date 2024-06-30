]#!/usr/bin/env bash
python -m tools.train --method=VAT --config=./config/exp_cfg/VAT.json --network=CSRNet --batch_size=10 --label_batch_size=5 --gpus=0