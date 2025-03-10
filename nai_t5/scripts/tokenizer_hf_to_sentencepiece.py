#!/usr/bin/env python

from pathlib import Path
import argparse
from nai_t5.sp_add_mask_vocab import add_mask_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tokenizer-in", type=Path, help="tokenizer input path", default="spiece.model")
    parser.add_argument("-c", "--tokenizer-config", type=Path, help="tokenizer config", default="tokenizer_config.json")
    parser.add_argument("-o", "--tokenizer-out", type=Path, help="tokenizer output path", default="spiece-ext.model")
    args = parser.parse_args()

    add_mask_vocab(
        tokenizer_in=args.tokenizer_in,
        tokenizer_config=args.tokenizer_config,
        tokenizer_out=args.tokenizer_out,
    )
    


if __name__ == "__main__":
    main()
