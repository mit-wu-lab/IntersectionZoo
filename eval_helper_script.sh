#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2022a

# Run the script
python -u code/policy_evaluation.py --dir wd/$EXP_NAME
