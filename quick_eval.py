import argparse
from typing import NamedTuple
from collections.abc import Sequence
from itertools import zip_longest
import json
import re
import os


class TSARItem(NamedTuple):
    context: str
    target: str
    candidates: Sequence[str]

    @staticmethod
    def from_components(
        context: str, target: str,
        candidates: Sequence[str],
        n_best: int | None,
        ignore_target: bool = True,
        ) -> 'TSARItem':
        if ignore_target:
            candidates = [c for c in candidates if c != target]
        return TSARItem(context=context, target=target, candidates=candidates[:n_best])

    @staticmethod
    def parse(line: str,
              n_best: int | None = None,
              ignore_target: bool = True
              ) -> 'TSARItem':
        context, target, *candidates = line.rstrip('\n').split('\t')
        return TSARItem.from_components(
            context, target, candidates, n_best, ignore_target
            )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('system_output', nargs='+', type=argparse.FileType('r'))
    parser.add_argument('--gold', type=argparse.FileType('r'))
    parser.add_argument('--decimals', '-d', type=int, default=4)
    parser.add_argument('--best', '-n', type=int, default=10)
    parser.add_argument('--diagnostics', '-D', action='store_true')
    parser.add_argument('--header', action='store_true')
    return parser.parse_args()


def score_div(a, b):
    if b == 0:
        assert a == 0
        return 1
    return a / b


def main() -> None:
    args = parse()
    n_best = args.best
    d = args.decimals

    if args.header:
        d_head = (
            '\tPrompt_Tokens\tCompletion_Tokens\tTotal_Tokens' if args.diagnostics else
            ''
            )
        print(
            f'Filename\tLanguage\t'
            f'Potential@{n_best}\tPrecision@{n_best}\tRecall@{n_best}\tF1@{n_best}\t'
            f'Accuracy@1\tAccuracy@1@top_gold_1{d_head}'
            )

    m = re.search('(/|^)Gold/([A-Za-z]+)/', args.gold.name)
    lang = '?' if m is None else m.group(2)

    # We use these in each pass => save as list:
    gold_lines = list(args.gold)

    for system_output in args.system_output:
        filename = system_output.name
        system_lines = list(system_output)

        pot_n = 0
        acc_n = 0
        acc_tg1_n = 0
        acc_pot_t = 0

        pre_rec_n = 0
        pre_t = 0
        rec_t = 0
        for i, (line_g, line_s) in enumerate(
            zip_longest(gold_lines, system_lines)
            ):

            if line_g is None or line_s is None:
                short_filename = args.gold.name if (line_g is None) else filename
                raise Exception(f'{short_filename} is shorter ({i} lines)).')

            gold    = TSARItem.parse(line_g)
            system  = TSARItem.parse(line_s, n_best=n_best)

            if gold[1] != system[1]:
                # TODO: We do not compare context, sice there are some small differences
                # in our gold data wrt. released test data (e.g. stripped whitespace)
                raise Exception(
                    f'Line {i+1}: Target differs between files:\n'
                    f'- gold: {gold[:2]}\n'
                    f'- system: {system[:2]}\n'
                    )

            gold_subs = set(gold.candidates)
            sys_subs = set(system.candidates)

            common = gold_subs.intersection(sys_subs)

            if common:
                pot_n += 1
            if system.candidates and system.candidates[0] in gold_subs:
                acc_n += 1
                if system.candidates[0] == gold.candidates[0]:
                    acc_tg1_n += 1

            pre_rec_n += len(common)
            rec_t += len(gold_subs)
            pre_t += len(sys_subs)
            acc_pot_t += 1

        pot = score_div(pot_n, acc_pot_t)
        pre = score_div(pre_rec_n, pre_t)
        rec = score_div(pre_rec_n, rec_t)
        f1 = 2 * (pre * rec) / (pre + rec) if (pre or acc) else 0.0
        acc = score_div(acc_n, acc_pot_t)
        acc_tg1 = score_div(acc_tg1_n, acc_pot_t)

        if args.diagnostics:
            path_diag = re.sub(r'(\.tsv)?$', '_diagnostics.json', system_output.name, 1)
            with open(path_diag) as df:
                diag = json.load(df)
            usage = diag['usage']
            assert len(usage) == 1
            t = usage.popitem()[1]
            pt = t['prompt_tokens']
            ct = t['completion_tokens']
            tt = t['total_tokens']
            d_str = f'\t{pt}\t{ct}\t{tt}'
        else:
            d_str = ''

        print(
            f'{os.path.basename(filename)}\t'
            f'{lang}\t'
            f'{pot:.{d}f}\t'
            f'{pre:.{d}f}\t'
            f'{rec:.{d}f}\t'
            f'{f1:.{d}f}\t'
            f'{acc:.{d}f}\t'
            f'{acc_tg1:.{d}f}{d_str}'
            )


if __name__ == '__main__':
    main()
