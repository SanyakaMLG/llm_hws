import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from evaluate import BASE_MODEL_ID, BASE_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION, dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
