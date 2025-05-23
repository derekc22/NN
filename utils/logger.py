import argparse
from datetime import datetime
# import yaml

parser = argparse.ArgumentParser(description="Log training runs")
parser.add_argument("--model", type=str, help="Specify model")
args = parser.parse_args()

dtn = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
fname = "logs/" + dtn + f"_{args.model}.log"

with open(f'config/{args.model}.yml', 'r') as config:
    config_contents = config.read()

with open(fname, 'w') as config_txt:
    config_txt.write(config_contents)



