#!/usr/bin/env python

from distutils.core import setup

setup(
    name='nai-t5',
    version='1.0',
    description='NovelAI T5',
    author='NovelAI/Anlatan Inc.',
    url='https://github.com/NovelAI/t5',
    license='Apache-2.0',
    packages=['nai_t5'],
    install_requires=[
        'torch',
        'einops',
        'pydantic',
    ],
    scripts=[
        'scripts/benchmark_attn.py',
        'scripts/benchmark_encoder.py',
        'scripts/t5_bucket_test.py',
        'scripts/t5_encdec_parity.py',
        'scripts/t5_encdec_precision_parity.py',
        'scripts/t5_encoder_flex_parity.py',
        'scripts/t5_encoder_hf_precision_parity.py',
        'scripts/t5_encoder_parity.py',
        'scripts/t5_encoder_parity_fsdp.py',
        'scripts/t5_encoder_precision_parity.py',
        'scripts/t5_flex_buckets.py',
        'scripts/t5_sampling_hf_generate.py',
        'scripts/t5_sampling_parity_cache.py',
        'scripts/t5_sampling_parity_nocache.py',
        'scripts/t5_serialize.py',
        'scripts/t5_serialize_dtensor.py',
        'scripts/tokenizer_hf_to_sentencepiece.py',
        'scripts/umt5_encdec_parity.py',
        'scripts/umt5_encoder_parity.py',
    ]
)