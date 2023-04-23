from dataclasses import dataclass
from typing import List
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
count_tokenizer = count_vectorizer.build_tokenizer()


@dataclass
class Params:
    model_name: str
    num_samples: int
    epochs: int
    batch_size: int
    path: str
    validation_pct: float
    max_seq_length: int
    data_augmentation: bool
    max_segment_length: int
    dataset_type: str
    augment_with_target_sentence_duplication: bool
    log_file_path: str


def save_results(params: Params, accuracy: List, val_accuracy: List, loss: List, val_loss: List):
    pass


# destructure the segments as they are currently in lists
def flatten(arr):
    return [item for sublist in arr for item in sublist]


def truncate_by_char(text, num_chars):
    return text[:num_chars]


def truncate_by_token(text, num_tokens):
    return " ".join(count_tokenizer(text)[:num_tokens])


def truncate_by_sentence(text, num_sentences):
    sentences = text.split(".")
    # add a trailing period only if truncation happens before the end.
    return ".".join(sentences[:num_sentences]) + (
        "." if num_sentences < len(sentences) else ""
    )


def avg_segment_length_by_char(segment, floor=True):
    return sum(map(len, segment)) // float(len(segment)) if floor == True else sum(map(len, segment)) / float(len(segment))


def avg_segment_length_by_token(segment, floor=True):
    return sum([len(word_tokenize(t)) for t in segment]) // float(len(segment)) if floor == True else sum([len(word_tokenize(t)) for t in segment]) / float(len(segment))


def dedupe_list(arr):
    return list(set(arr))

def truncate_string(text, max_chars=30):
    truncated_string = (
            (text[:max_chars] + "..") if len(text) > max_chars else text
        )
    return truncated_string
