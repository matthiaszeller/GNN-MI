

import argparse
import json

import wandb

import sys
sys.path.insert(0, '../src')

import setup
import utils


# User input
parser = argparse.ArgumentParser()
parser.add_argument('run_id')
args = parser.parse_args()

# Wandb fetch run config
config = utils.get_run_config(args.run_id)

# Write config in file
output_file = f'config-{args.run_id}.json'

with open(output_file, 'w') as f:
    json.dump(config, f, indent=4)


