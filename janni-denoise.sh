#!/usr/bin/env bash
export PYTHONPATH=/home/tjr22/relion-external:$PYTHONPATH
/net/prog/anaconda3/envs/cryolo/bin/python /home/tjr22/relion-external/JANNI/denoise.py "$@"
