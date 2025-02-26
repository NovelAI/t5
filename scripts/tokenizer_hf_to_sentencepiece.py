#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import List

import sentencepiece.sentencepiece_model_pb2 as spm


# based on (Apache-licensed):
# https://github.com/google/sentencepiece/blob/master/python/add_new_vocab.ipynb
# note: this also seems to agree with Google's approach in SeqIO, so I think we're on the right track:
# https://github.com/google/seqio/blob/4d3097973e9e24ec2963319ec3c5ff518811060f/seqio/vocabularies.py#L362
# https://github.com/huggingface/transformers/pull/24565
def main():
    """
    Takes an existing HF-style spiece.model (which lacks masked vocabulary),
    Reads the masked tokens from the HF tokenizer config,
    Creates a new spiece.model which includes that vocabulary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tokenizer-in", type=Path, help="tokenizer input path", default="spiece.model")
    parser.add_argument("-c", "--tokenizer-config", type=Path, help="tokenizer config", default="tokenizer_config.json")
    parser.add_argument("-o", "--tokenizer-out", type=Path, help="tokenizer output path", default="spiece-ext.model")
    args = parser.parse_args()

    m = spm.ModelProto()
    with open(args.tokenizer_in, "rb") as f:
        m.ParseFromString(f.read())
    print(f"Loaded {args.tokenizer_in}.\nInitial vocab size is: {len(m.pieces)}")

    with open(args.tokenizer_config, "rb") as f:
        tok_config = json.load(f)

    special_toks: List[str] = tok_config["additional_special_tokens"]

    # HF T5 config lists the <extra_id_*> additional_special_tokens from 0 to 99, but they need to be appended from 99 to 0
    special_toks = list(reversed(special_toks))

    tok_str: str = "\n".join(special_toks)
    print(f"Adding {len(special_toks)} special tokens to vocab, namely:\n{tok_str}.")

    for tok in special_toks:
        tok_sp = m.SentencePiece()
        tok_sp.piece = tok
        tok_sp.score = 0
        # https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
        # https://github.com/google/sentencepiece/blob/7dcb541451b1862d73f473b3804ccf8f2a9e10f6/src/builtin_pb/sentencepiece_model.pb.cc#L231-L236
        # User defined symbol is handled as one piece in any context. If this symbol is included in the input text, this symbol is always extracted as one piece.
        tok_sp.type = tok_sp.USER_DEFINED
        m.pieces.append(tok_sp)
    print(f"Appended {len(special_toks)} special tokens.\nVocab size is now: {len(m.pieces)}")

    with open(args.tokenizer_out, "wb") as f:
        f.write(m.SerializeToString())
    print(f"Wrote tokenizer to {args.tokenizer_out}.")


if __name__ == "__main__":
    main()
