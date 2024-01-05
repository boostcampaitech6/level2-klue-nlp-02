import argparse

from inference import predict
from train import train


def main(args):
    train()
    predict(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_model")
    args = parser.parse_args()
    print(args)
    main(args)
