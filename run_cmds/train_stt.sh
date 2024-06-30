]#!/usr/bin/env bash
python -m tools.train --method=STT --config=./config/exp_cfg/STT.json --network=CSRNet --batch_size=16 --label_batch_size=8 --gpus=0