from utils.experiments import (
    get_experiments_json,
    get_experiments,
    save_results,
    print_break,
)
import numpy as np
import pandas as pd
import json
import toml
from tensorflow.python import keras
from src.dataset.ldabertv3 import LDABERT3Dataset
from src.encoders.context_encoder_ldabert_2 import ContextEncoderSimple
import nltk
import tensorflow
import sentence_transformers
import transformers
import sys
import os
import config
import csv

import tensorflow as tf
import random
import string

from utils.metrics import windowdiff, pk
from collections import OrderedDict

from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, config.root_path)


def get_random_hash(k):
    x = "".join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1
    return (TP, FP, TN, FN)


def get_precision(TP, FP, TN, FN):
    if TP == 0:
        return 0
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def get_recall(TP, FP, TN, FN):
    if TP == 0:
        return 0
    if TP + FP == 0:
        return 0
    return TP / (TP + FN)


@dataclass
class CoherenceExperiment:
    model_string: Optional[str] = "bert-base-uncased"
    max_words_per_step: Optional[int] = 4
    same_word_multiplier: Optional[int] = 2
    no_same_word_penalty: Optional[int] = 1
    prediction_threshold: Optional[float] = 0.25
    coherence_dump_on_prediction: Optional[bool] = False
    pruning: Optional[int] = 1  # remove one sentence worth of keywords
    pruning_min: Optional[
        int
    ] = 6  # remove the first sentence in the coherence map once it grows passed 6
    dynamic_threshold: Optional[bool] = False
    threshold_warmup: Optional[
        int
    ] = 10  # number of iterations before using dynamic threshold
    last_n_threshold: Optional[
        int
    ] = 5  # will only consider the last n thresholds for dynamic threshold


class SimpleExperiment:
    def __init__(self):
        self.experiments = []

    def queue_experiment(self, experiment: CoherenceExperiment):
        self.experiments.append(experiment)

    def get_experiments(self):
        return self.experiments
