#!/usr/bin/env bash
python -m tools.train --method=MT --config=./config/exp_cfg/MT.json --network=CSRNet --batch_size=12 --label_batch_size=6 --gpus=0