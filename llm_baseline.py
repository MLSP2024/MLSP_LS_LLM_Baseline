'''
Copyright (c) 2024, Adam Nohejl
'''
import os
import re
import argparse
import json
from tqdm import tqdm
from typing import Optional
from collections import defaultdict, Counter
from collections.abc import Sequence, Iterable
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from cleanup import clean_predictions


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


QUANTIZATION2TORCH_DTYPE = {
    'auto': 'auto',
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
    }

MAX_PREDICTIONS = 10

LANG2FULL_NAME: dict[str, Optional[str]] = {
    'en': None,
    'ca': 'Catalan',
    'fil': 'Filipino',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'si': 'Sinhala',
    'es': 'Spanish'
    }


def lang_for_prompt(lang: str, fil_tagalog: bool) -> Optional[str]:
    return (
        'Tagalog' if ((lang == 'fil') and fil_tagalog) else
        LANG2FULL_NAME[lang]
        )


def lang_for_filename(lang: str) -> str:
    return LANG2FULL_NAME[lang] or 'English'


def initialize_model(
    model_name, device, quantization, huggingface_token=None
    ):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=huggingface_token if huggingface_token else None,
        trust_remote_code=True,
        padding_side='left'  # For optimal performance with batching
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        'token': huggingface_token if huggingface_token else None,
        'low_cpu_mem_usage': True
        }

    if quantization == '4bit':
        model_kwargs.update({
            'quantization_config': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                ),
            'torch_dtype': torch.float16
            })
    elif quantization == '8bit':
        model_kwargs.update({
            'device_map': 'auto',
            'trust_remote_code': True,
            'load_in_8bit': True,
            'use_cache': True
            })
    elif quantization == 'half':
        pass
    elif quantization in QUANTIZATION2TORCH_DTYPE:
        # helpful for e.g. https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b
        model_kwargs.update({
            'torch_dtype': QUANTIZATION2TORCH_DTYPE[quantization],
            })
    elif quantization == 'none':
        pass
    else:
        raise ValueError(
            'Invalid quantization. Use "none", "half", "auto", "bfloat16", '
            '"float16", "8bit", or "4bit".'
            )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if quantization == 'half':
        model = model.half()

    if quantization not in ['4bit', '8bit']:
        model.to(device)

    model.eval()
    return model, tokenizer


def get_params(prompt_name: str) -> dict:
    return dict(do_sample=True, temperature=0.3, top_p=1)


def get_prompt(
    prompt_name: str, context: str, word: str,
    lang: Optional[str] = 'en', fil_tagalog: bool = False
    ) -> str:
    lang_name = lang_for_prompt(lang, fil_tagalog)
    lang_space = '' if (lang_name is None) else f'{lang_name} '

    if prompt_name == 'unihd':
        return (
            f'Context: {context}\n'
            f'Question: Given the above context, list ten alternative {lang_space}'
            f'words for \"{word}\" that are easier to understand.\n'
            f'Answer:'
            )
    if prompt_name == 'only':
        return (
            f'Context: {context}\n'
            f'Question: Given the above context, list ten alternative {lang_space}'
            f'words for \"{word}\" that are easier to understand. '
            f'List only the words without translations, transcriptions '
            f'or explanations.\n'
            f'Answer:'
            )

def count_tokens(tokens_ids, tokenizer, count_eos):
    n = len(tokens_ids)
    n_tokens = n
    # Skip padding:
    while (
        n_tokens >= 1 and
        tokens_ids[n_tokens - 1] == tokenizer.pad_token_id or
        tokens_ids[n_tokens - 1] == 0  # unk_token in padded generated sequences
        ):
        n_tokens -= 1

    n_pad_left = 0
    for n_pad_left in range(n_tokens):
        if tokens_ids[n_pad_left] != tokenizer.pad_token_id:
            break

    # Unskip 1 EOS token if it's there:
    if count_eos:
        assert tokenizer.pad_token_id == tokenizer.eos_token_id
        if n_tokens < n and tokens_ids[n_tokens] == tokenizer.eos_token_id:
            n_tokens += 1

    return n_tokens - n_pad_left


def batched(xs: Sequence, k=1) -> Iterable[Sequence]:
    n = len(xs)
    for i in range(0, n, k):
        yield xs[i:i + k]


def main(args):
    print(f'Running model with quantization type: {args.quantization}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size
    model, tokenizer = initialize_model(
        args.model, device, args.quantization, args.huggingface_token
        )

    prompt_name = args.prompt_name
    # TODO lang lang, output_names
    input_files = args.input_files
    output_files = args.output_files

    if not output_files:
        output_files = ['output.tsv'] if (len(input_files) == 1) else ['output']
    if len(output_files) == 1 and len(input_files) > 1:
        output_dir = output_files[0]
        os.makedirs(output_dir, exist_ok=True)
        output_files = [
            os.path.join(
                output_dir,
                re.sub(r'_unlabelled|_labels', '', os.path.basename(path))
                )
            for path in input_files
            ]

    assert len(input_files) == len(output_files), (len(input_files), len(output_files))

    seed = args.seed
    tagalog = args.tagalog

    params = get_params(prompt_name)

    fmt_combined = False
    for path_in, path_out in zip(input_files, output_files):
        lang = args.language
        if lang is None:
            if '_combined_' in path_in:
                fmt_combined = True
            else:
                m = re.match(r'(^|.*/)multilex_(test|trial)_([a-z]+)_', path_in)
                if m is None:
                    raise Exception(
                        'Cannot detect language from filename. Use --language.'
                        )
                lang = m.group(3)
                assert 2 <= len(lang) <= 3, f'Unexpected language: {lang}'
        with open(path_in, 'r') as fi, open(path_out, 'w') as fo:
            lines = fi.readlines()[:args.limit]
            usage_by_lang = defaultdict(Counter)
            inference_time = 0
            errors = []
            diagnostics = {
                'usage': usage_by_lang,
                'model': args.model,
                'prompt_name': prompt_name,
                'tagalog': tagalog,
                'quantization': args.quantization,
                'batch_size': batch_size,
                'seed': seed,
                'errors': errors
                }

            for batch_idx, batch_lines in enumerate(
                tqdm(list(batched(lines, batch_size)))
                ):
                contexts = []
                words = []
                prompts = []
                # Extract context and complex word
                for line in batch_lines:
                    if fmt_combined:
                        lang_idx, context, word, *_ = line.strip('\n ').split('\t')
                        lang, _idx_str = lang_idx.split('_', 1)
                        assert 2 <= len(lang) <= 3, f'Unexpected language: {lang}'
                    else:
                        context, word, *_ = line.strip('\n ').split('\t')
                        assert lang is not None
                    contexts.append(context)
                    words.append(word)
                    prompt = get_prompt(
                        prompt_name, context, word,
                        lang, fil_tagalog=tagalog
                        )
                    prompts.append(prompt)

                set_seed(seed)

                prompts = [
                    # BOS will be added when tokenizing, so we remove it:
                    tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': prompt}],
                        tokenize=False
                        ).removeprefix(tokenizer.bos_token)
                    for prompt in prompts
                    ]
                inputs = tokenizer(
                    prompts, return_tensors='pt', padding=True,
                    truncation=True, max_length=args.input_max_length
                    )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                max_input_tokens = len(inputs['input_ids'][0])  # padded to max
                ns_input_tokens = [
                    count_tokens(
                        input_sequence, tokenizer, count_eos=False
                        )
                    for input_sequence in inputs['input_ids']
                    ]

                if device.startswith('cuda'):
                    torch.cuda.synchronize(device=device)

                start_time = time.perf_counter()
                with torch.no_grad():
                    generated_outputs = model.generate(
                        **inputs,
                        # Note: max_length = in + out
                        max_new_tokens=args.output_max_length,
                        **params
                        )
                end_time = time.perf_counter()
                inference_time += (end_time - start_time)

                n_output_tokens = 0
                for i, (
                    generated_outputs_i, context, word
                    ) in enumerate(
                    zip(generated_outputs, contexts, words)
                    ):
                    # Generated output contains input, skip it:
                    output_tokens = generated_outputs_i[max_input_tokens:]
                    n_output_tokens_i = count_tokens(
                        output_tokens, tokenizer, count_eos=True
                        )
                    n_output_tokens += n_output_tokens_i

                    output_text = tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                        )

                    clean_error = None
                    try:
                        predictions = clean_predictions(output_text, word)
                    except Exception as e:
                        clean_error = e
                        errors.append(str(e))
                        predictions = []

                    prediction_string = '\t'.join(predictions[:MAX_PREDICTIONS])

                    if args.verbose or (clean_error is not None):
                        idx = batch_idx * batch_size + i
                        print()
                        print('--------------------------')
                        print(f'* idx: {idx} context: {context} target: {word}')
                        print(f'* output: {output_text}')
                        print(f'* prompt_tokens: {n_output_tokens}')
                        print(f'* completion_tokens: {n_output_tokens_i}')
                        print(f'* predictions: {predictions}')
                        print(f'* time/instance: '
                              f'{(end_time - start_time) / len(words):.3f} s')
                        if clean_error:
                            print(f'* ERROR in clean_predictions(): {clean_error}')

                    fo.write(f'{context}\t{word}\t{prediction_string}\n')

                    n_input_tokens = sum(ns_input_tokens)
                    usage_by_lang[lang].update({
                        'prompt_tokens': n_input_tokens,
                        'completion_tokens': n_output_tokens,
                        'total_tokens': n_input_tokens + n_output_tokens
                        })

        if args.verbose:
            print(f'inference time:          {inference_time:.3f} s')
            print(f'inference time/instance: {inference_time/len(lines):.3f} s')

        diagnostics['inference_time'] = inference_time
        path_diag = re.sub(r'(\.tsv)?$', '_diagnostics.json', path_out, 1)
        with open(path_diag, 'w') as f:
            json.dump(diagnostics, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-70b-chat-hf', help=(
        'Model identifier on the Huggingface Hub '
        '(default: meta-llama/Llama-2-70b-chat-hf).'
        ))
    parser.add_argument('--tagalog', action='store_true',
                        help='Call Filipino "Tagalog" in prompts.')
    parser.add_argument(
        'input_files', nargs='*',
        default=[
            'MLSP_Data/Data/Test/All/multilex_test_all_combined_ls_unlabelled.tsv'
            ],
        help='Input files in LS format (default all test data).'
        )
    parser.add_argument('--output-files', '-o', nargs='*', default=[], help=(
        'Output files in LCP format. If single name is given for multiple input files, '
        'it is understood as a directory name for multiple files. '
        'Default: output (output.tsv).'
        ))

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--language', type=str,
                        help='Language (default: automatically from filename).')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for generation.')
    parser.add_argument(
        '--quantization',
        choices=['none', 'half', '8bit', '4bit', 'auto', 'bfloat16', 'float16'],
        default='4bit', help='Type of quantization.'
        )
    parser.add_argument(
        '--huggingface-token', type=str, default=True,
        help='Hugging Face token for gated models (default: true => read from config).'
        )
    parser.add_argument('--prompt-name', default='only')
    parser.add_argument('--input-max-length', type=int, default=512,
                        help='Maximum sequence length for the tokenizer.')
    parser.add_argument('--output-max-length', type=int, default=256,
                        help='Maximum length for the generated text.')
    parser.add_argument('--limit', '-n', type=int, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    main(args)
