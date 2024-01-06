import yaml

from inference import predict
from train import train


def main(configs):
    train(configs)
    predict(configs)


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        configs = yaml.safe_load(f)

    main(configs)
