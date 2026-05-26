#!/bin/bash
set -e

python experiments/defense_eval.py --device auto --epochs 200
