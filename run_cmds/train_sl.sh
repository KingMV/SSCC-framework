#!/usr/bin/env bash
python -m tools.train --method=SL --config=./config/exp_cfg/SL.json --network=CSRNet --batch_size=12 --label_batch_size=12 --gpus=1