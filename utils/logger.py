import argparse
from datetime import datetime
import yaml
# from ruamel.yaml import YAML
import os


def load_config(yaml_path):
  with open(yaml_path, "r") as f:
    return yaml.safe_load(f)

  
# ruamel_yaml = YAML()
# ruamel_yaml.preserve_quotes = True 
# def load_ruamel_config(yaml_path):
#   with open(yaml_path, "r") as f:
#     return ruamel_yaml.load(f)
  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Log training runs")
    parser.add_argument("--model", type=str, help="Specify model")
    args = parser.parse_args()

    dtn = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_folder_name = "logs/" + dtn + f"_{args.model}"
    log_file_name = log_folder_name + "/log.log"
    os.makedirs(log_folder_name, exist_ok=True)
    
    yaml_path = f"config/{args.model}.yml"
    config = load_config(yaml_path)
    config['log_id'] = log_folder_name

    with open(yaml_path, 'w') as file: 
        yaml.dump(config, file, default_flow_style=False)

    # with open(f'config/{args.model}.yml', 'r') as config:
        # config_contents = config.read()

    with open(log_file_name, 'w') as config_txt:
        config_txt.write(yaml.dump(config, default_flow_style=False))




