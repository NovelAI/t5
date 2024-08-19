import argparse
from pathlib import Path
from typing import Dict, OrderedDict

from torch import Tensor
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

# TODO: bring some kind of save function back
# from basedformer.lm_utils import save
from nai_t5 import T5, T5Config, hf_to_based_t5_state, to_based_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="HF model name. example: google/t5-v1_1-small"
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="output directory. example: /mnt/clusterstorage/models/t5-goog/t5-v1_1-small",
    )
    args = parser.parse_args()

    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(args.model_name)
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(args.model_name, legacy=False)
    hf_t5 = T5ForConditionalGeneration.from_pretrained(args.model_name).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    my_t5 = T5(my_config).eval()

    hf_state: OrderedDict[str, Tensor] = hf_t5.state_dict()
    converted_enc_state: Dict[str, Tensor] = hf_to_based_t5_state(hf_state, my_config)
    my_t5.load_state_dict(converted_enc_state)

    # save(my_t5, str(args.out_dir))


if __name__ == "__main__":
    main()
