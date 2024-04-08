'''
Based on UniHD cleanup:
<https://github.com/dennlinger/TSAR-2022-Shared-Task/blob/main/context_predictor.py>
Copyright (c) 2022, Dennis Aumiller: <https://aclanthology.org/2022.tsar-1.28>
Copyright (c) 2024, Adam Nohejl: modifications for BEA 2024 MLSP Shared Task baseline
'''

import sys
import regex
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

ITEM_SEPARATORS = ',、，'  # Added for Japanese/Chinese: 、，


def clean_predictions(text: str, given_word: str, lang: str = 'en') -> List[str]:
    """
    Post-processing of files, by trying different strategies to coerce it into actual
    singular predictions.
    :param text: Unfiltered text predicted by a language model
    :param given_word: The word that is supposed to be replaced. Sometimes appears in
    `text`.
    :return: List of individual predictions
    """

    # Catch sample 248
    if text.startswith(given_word):
        text = text[len(given_word):]

    # Clear additional clutter that might have been encountered
    text = text.strip("\n :;.?!")

    # Added for Llama2, which returns empty text in rare cases:
    if not text:
        return []

    # We first split lines (whether there are newlines or not) to handle all cases
    # in an an uniform way. Possible cases:
    # - one simplification per line: any "explanations" before/after are discarded
    #   later using remove_multiwords/remove_long
    # - single line with multiple simplifications: any "explanations" before/after are
    #   discarded in the following loop

    lines = list(filter(None, text.split('\n')))  # non-empty lines
    cleaned_predictions = []
    for line in lines:
        line = line.strip(':;.?!')
        # Choose the most common separator:
        [(top_sep, top_sep_n)]  = Counter({
            s: line.count(s) for s in ITEM_SEPARATORS
            }).most_common(1)
        preds = []
        if top_sep_n >= 5:  # 5 separators => at least 6 predictions
            preds = line.split(top_sep)
        elif contains_numerals(line):   # Generally requires at least #1 and #6
            preds = regex.split(RE_NUMERAL, line)
        elif top_sep_n:  # Fallback on separators even if there are just 1 or 2 of them:
            preds = line.split(top_sep)
        if len(preds) >= len(cleaned_predictions):
            cleaned_predictions = preds

    if len(lines) > len(cleaned_predictions):
        # Interpret lines as predictions:
        cleaned_predictions = lines

    if len(cleaned_predictions) < 5:
        sys.stderr.write(
            f"Unclear format: '{text}'.\n"
            f"Parsing as {cleaned_predictions}.\n"
            )

    # Remove numerals
    cleaned_predictions = [remove_numerals(pred) for pred in cleaned_predictions]
    # Make sure everything is lower-cased and stripped
    cleaned_predictions = [pred.lower().strip(" \n") for pred in cleaned_predictions]
    # Remove "to" in the beginning
    cleaned_predictions = [remove_to(pred) for pred in cleaned_predictions]
    # Remove predictions that match the given word
    cleaned_predictions = remove_identity_predictions(cleaned_predictions, given_word)
    # Remove empty predictions that may have slipped through:
    cleaned_predictions = remove_empty_predictions(cleaned_predictions)
    # Remove trailing text in parentheses (Japanese and Chinese)
    cleaned_predictions = [remove_paren(pred) for pred in cleaned_predictions]
    # Remove trailing text in parentheses (Llama2 English)
    cleaned_predictions = [remove_explanation(pred) for pred in cleaned_predictions]
    # Remove examples, quotes (Llama2 Japanese, Portuguese, ...)
    cleaned_predictions = [remove_example(pred) for pred in cleaned_predictions]
    cleaned_predictions = [remove_quotes(pred) for pred in cleaned_predictions]

    # Remove punctuation
    cleaned_predictions = remove_punctuation(cleaned_predictions)

    # Remove multi-word/long predictions (language-specific)
    if lang == 'ja':
        cleaned_predictions = remove_long(cleaned_predictions, max_chars=10)
    elif lang == 'zh':
        cleaned_predictions = remove_long(cleaned_predictions, max_chars=4)
    elif lang == 'en':
        cleaned_predictions = remove_multiwords(cleaned_predictions, max_segments=2)
    else:
        cleaned_predictions = remove_multiwords(cleaned_predictions, max_segments=3)

    return cleaned_predictions


def remove_punctuation(predictions: List[str]) -> List[str]:
    return [prediction.strip(".,;?!") for prediction in predictions]


def remove_multiwords(predictions: List[str], max_segments: int = 3) -> List[str]:
    return [
        prediction for prediction in predictions
        if len(prediction.split(" ")) <= max_segments
        ]


def remove_long(predictions: List[str], max_chars: int) -> List[str]:
    return [prediction for prediction in predictions if len(prediction) <= max_chars]


def remove_empty_predictions(predictions: List[str]) -> List[str]:
    return [pred for pred in predictions if pred.strip("\n ")]


def remove_identity_predictions(predictions: List[str], given_word: str) -> List[str]:
    return [pred for pred in predictions if pred != given_word]


NUMERAL_EXAMPLES = (
    ('1.', '6.'),
    ('(1)', '(6)'),
    ('a)', 'f)'),
    ('👉')
    )


def contains_numerals(s: str) -> bool:
    return any(
        all(ex in s for ex in exs)
        for exs in NUMERAL_EXAMPLES
        )


RE_NUMERAL = r'(?:\([0-9]{1,2}\)|[0-9]{1,2}\.?|[a-z]\)|👉)'  # non-capturing (for split)


def remove_numerals(text: str) -> str:
    """
    Will remove any leading numerals etc.
    :param text: Input text, potentially containing a leading numeral
    :return: cleaned text
    """

    # Added (1) (2) (3) for LLama2: \([0-9]{1,2}\)
    # Added a) b) c) for LLama2: [a-z]\)
    # Added 👉 for Llama2, - for ELYZA-Llama2

    return regex.sub(rf'^ *({RE_NUMERAL}|-) ?', '', text)


def remove_to(text: str) -> str:
    """
    Removes the leading "to"-infinitive from a prediction, which is sometimes caused
    when the context word is preceeded with a "to" in the text.
    :param text: Prediction text
    :return: Text where a leading "to " would be removed from the string.
    """
    return regex.sub(r"^to ", "", text)


def remove_paren(text: str) -> str:
    '''
    Remove readings or translations in parentheses, which occur almost regularly in
    case of Japanese and Chinese, e.g. '憂鬱な (ゆううつな)', '難しい (difficult)'.
    We ignore that in rare cases the pattern is reversed ('difficult (難しい)').
    '''
    return regex.sub(r' *\(.*$', '', text)


def remove_explanation(text: str) -> str:
    '''
    Remove explanation separated by a hyphen, which occur often in Llama2 predictions,
    e.g.: 'Downfall - The fall or defeat of a ruler or government.'
    '''
    return regex.sub(r' - .*$', '', text)


def remove_quotes(text: str) -> str:
    '''
    Remove quotes, which occur often in Llama2 predictions.'
    '''
    return regex.sub(r'"(.*)"', r'\1',
                     regex.sub(r'「(.*)」', r'\1', text))


def remove_example(text: str) -> str:
    '''
    Remove examples, which somtimes often in Llama2 predictions, e.g.:
    'Example: 彼の大胆なプロポーズが彼女の心を捉えた (His bold proposal captured her heart.)'

    If the whole line is an example, the empty string will be removed in further steps.
    '''
    return regex.sub(r'\bexample: .*$', '', text)   # already lower-cased


def deduplicate_predictions(predictions: List[Tuple]) -> Dict:
    """
    Slightly less efficient deduplication method that preserves "ranking order" by
    appearance.
    :param predictions: List of predictions
    :return: Filtered list of predictions that no longer contains duplicates.
    """
    merged = defaultdict(float)
    for prediction, score in predictions:
        merged[prediction] += score

    return merged
