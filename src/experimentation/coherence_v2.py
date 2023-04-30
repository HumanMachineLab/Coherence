import config, os, sys, random, string, csv
import pandas as pd

from dataclasses import dataclass, asdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, config.root_path)


from db.dbv2 import Table, AugmentedTable, TrainTestTable
from src.dataset.utils import flatten, dedupe_list, truncate_string
from src.encoders.coherence import Coherence
from src.experimentation.prediction_thresholds import thresholds

from utils.metrics import windowdiff, pk


def get_random_hash(k):
    x = "".join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


experiment_set_hash = get_random_hash(
    5
)  # global hash for all the experiments during this run.


@dataclass
class CoherenceExperiment:
    num_samples: int = 50  # how many samples to test on
    start: int = 0  # where to start the predictions from in the dataset
    dataset_type: str = "city"  # either city or disease
    model_string: str = "bert-base-uncased"
    max_words_per_step: int = 4
    same_word_multiplier: int = 2
    no_same_word_penalty: int = 1
    prediction_threshold: float = 0.25
    coherence_dump_on_prediction: bool = False
    pruning: int = (
        1  # remove one sentence worth of keywords. Set to 0 if pruning is not desired
    )
    pruning_min: int = (
        6  # remove the first sentence in the coherence map once it grows past 6
    )
    dynamic_threshold: bool = False
    threshold_warmup: int = 10  # number of iterations before using dynamic threshold
    last_n_threshold: int = (
        5  # will only consider the last n thresholds for dynamic threshold
    )
    kb_embeddings: bool = False  # whether to use the built in keybert embeddings (less accurate, but faster)
    experiment_hash: str = None  # a unique identifier for the experiment


class SimpleExperiment:
    def __init__(self):
        self.experiments = []

    def queue_experiment(self, experiment: CoherenceExperiment):
        experiment.experiment_hash = get_random_hash(5)
        self.experiments.append(experiment)

    def get_experiments(self):
        return self.experiments

    # gather all the predictions and store them in a csv.
    def log_predictions(
        self, experiment: CoherenceExperiment, predictions, true_labels
    ):
        predictions_directory = "{}/predictions/{}".format(
            config.root_path, experiment_set_hash
        )

        if not os.path.exists(predictions_directory):
            os.makedirs(predictions_directory)

        predictions_path = "{}/predictions.csv".format(predictions_directory)

        # don't add headers to the file if it already exists.
        hdr = False if os.path.isfile(predictions_path) else True

        # turn the experiment dataclass into a dictionary so we can put it into a csv file
        experiment_dict = asdict(experiment)

        # add our experiment data to the predictions csv
        experiment_dict["logits"] = [
            x[0].item() for x in predictions
        ]  # convert pytorch tensor to raw primitive type before storage
        experiment_dict["predictions"] = [x[1] for x in predictions]
        experiment_dict["true_labels"] = true_labels

        df = pd.json_normalize(experiment_dict)

        # append data frame to CSV file
        df.to_csv(predictions_path, mode="a", index=False, header=hdr)

        self.log_evaluations(experiment, predictions, true_labels)

    # calculate all the evaluation metrics and store them in a csv file.
    def log_evaluations(self, experiment, predictions, true_labels):
        # get the thresholds from a map pertaining to the current model
        curr_model_thresholds = thresholds[experiment.model_string]

        evaluation_directory = "{}/predictions/{}/{}".format(
            config.root_path, experiment_set_hash, experiment.experiment_hash
        )

        if not os.path.exists(evaluation_directory):
            os.makedirs(evaluation_directory)

        evaluation_file = "{}/eval.csv".format(evaluation_directory)

        # don't add headers to the file if it already exists.
        hdr = False if os.path.isfile(evaluation_file) else True

        df_data = []

        for pred_thresh in curr_model_thresholds:
            # calculate the predictions based on the current threshold
            modified_predictions = [
                1 if x < pred_thresh else 0 for x in [x[0] for x in predictions]
            ]

            avg_k = len(true_labels) // (
                true_labels.count(1) + 1
            )  # get avg segment size

            # convert to strings so it can be used with the wd and pk metrics
            pred_string = "".join(str(modified_predictions))
            true_string = "".join(str(true_labels))

            # calculate all the metrics we will be storing
            wd_score = windowdiff(pred_string, true_string, avg_k)
            pk_score = pk(pred_string, true_string, avg_k)
            tn, fp, fn, tp = confusion_matrix(true_labels, modified_predictions).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, modified_predictions, average="macro"
            )

            # append all the data to an array before converting to a dataframe below
            df_data.append(
                [
                    pred_thresh,
                    avg_k,
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    pk_score,
                    wd_score,
                ]
            )

        df_evaluation_set = pd.DataFrame(
            df_data,
            columns=[
                "prediction_threshold",
                "K",
                "TP",
                "FP",
                "TN",
                "FN",
                "precision",
                "recall",
                "f1",
                "Pk",
                "WindowDiff",
            ],
        )

        # append data frame to CSV file
        df_evaluation_set.to_csv(evaluation_file, mode="a", index=False, header=hdr)

    # run the actual experiment
    def run(self):
        print(f"Running experiment set: {experiment_set_hash}")
        for i, experiment in enumerate(self.experiments):
            print(f"Running experiment: {experiment}")
            table = Table(experiment.dataset_type)

            all_segments = table.get_all_segments()

            segments = [[y[1] for y in x] for x in all_segments]
            segments_labels = [
                [1 if i == 0 else 0 for i, y in enumerate(x)] for x in all_segments
            ]

            flattened_segments = flatten(segments)
            flattened_labels = flatten(segments_labels)

            segments_to_test = flattened_segments[
                experiment.start : experiment.start + experiment.num_samples
            ]
            labels_to_test = flattened_labels[
                experiment.start : experiment.start + experiment.num_samples
            ]

            # initialize the coherence library
            coherence = Coherence(
                max_words_per_step=experiment.max_words_per_step,
                same_word_multiplier=experiment.same_word_multiplier,
                no_same_word_penalty=experiment.no_same_word_penalty,
                model_string=experiment.model_string,
            )

            logits = coherence.predict(
                text_data=segments_to_test,
                max_tokens=256,
                prediction_threshold=experiment.prediction_threshold,
                pruning=experiment.pruning,
                pruning_min=experiment.pruning_min,
                coherence_dump_on_prediction=False,
            )

            self.log_predictions(experiment, logits, labels_to_test)

            print(f"\nExperiment {i+1} complete.")
            print("==============================================\n")
