import yaml
import argparse
import pandas as pd


def read_params(config_path):
    with open(config_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def load_data(data_path, model_var):
    df = pd.read_csv(data_path, sep=',', encoding='utf-8')
    df = df[model_var]
    return df


def load_raw_data(config_path):
    config = read_params(config_path)
    external_data_path = config['external_data_config']['external_data_csv']
    raw_data_path = config['raw_data_config']['raw_data_csv']
    model_var = config['raw_data_config']['model_var']

    df = load_data(external_data_path, model_var)
    df.to_csv(raw_data_path, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yml')
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)
