#!/bin/bash

mutations=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

for scale in "${mutations[@]}"
do
  echo "Running with scale_mutation = $scale"
  python3 codes/main_map_elite_action.py --render_mode False \
      --generation_mode True \
      --stock_path data/analysis_qd_action \
      --scale_mutation "$scale"
done
