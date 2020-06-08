#!/usr/bin/env bash
export PYTHONPATH=/home/tjr22/relion-external:$PYTHONPATH
/net/prog/anaconda3/envs/topaz/bin/python /home/tjr22/relion-external/Topaz/pick_train.py "$@"