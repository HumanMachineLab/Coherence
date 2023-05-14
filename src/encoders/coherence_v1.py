import torch
from src.bertkeywords.src.similarities import Embedding, Similarities
from src.bertkeywords.src.keywords import Keywords
from src.dataset.utils import dedupe_list, flatten, truncate_string, truncate_by_token
import time

supported_models = [
    "sentence-transformers/LaBSE",
    "bert-base-uncased",
    "roberta-base",
    "sentence-transformers/all-MiniLM-L6-v2",
]


class Coherence:
    def __init__(
        self,
        max_words_per_step=2,
        coherence_threshold=0.4,
        same_word_multiplier=2,  # if set to 1, don't amplify the same words found
        no_same_word_penalty=1,  # if set to 1, don't penalize for not finding the same word.
        model_string="bert-base-uncased",
        kb_embeddings=False,  # if set to True, use the keybert embeddings.
    ):
        self.max_words_per_step = max_words_per_step
        self.coherence_threshold = coherence_threshold
        self.same_word_multiplier = (
            same_word_multiplier  # if set to 1, don't amplify the same words found
        )
        self.no_same_word_penalty = no_same_word_penalty  # if set to 1, don't penalize for not finding the same word.
        self.model_string = model_string
        self.kb_embeddings = kb_embeddings

        if model_string not in supported_models:
            self.model_string = "bert-base-uncased"

        similarities_lib = Similarities(self.model_string)

        self.keywords_lib = Keywords(similarities_lib.model, similarities_lib.tokenizer)
        self.embedding_lib = Embedding(
            similarities_lib.model, similarities_lib.tokenizer
        )

    def get_similar_coherent_words(
        self, prev_sentence, curr_sentence, coherence_threshold
    ):
        if self.kb_embeddings:
            embedding_technique = self.keywords_lib.get_keywords_with_embeddings
        else:
            embedding_technique = self.keywords_lib.get_keywords_with_kb_embeddings

        kw_curr_sentence = embedding_technique(curr_sentence)[: self.max_words_per_step]
        kw_prev_sentence = embedding_technique(prev_sentence)[: self.max_words_per_step]

        coherent_words = []

        for word2 in kw_curr_sentence:
            for word1 in kw_prev_sentence:
                # check similarity and add to coherent dictionary
                emb1 = word1[2]
                emb2 = word2[2]
                similarity = torch.cosine_similarity(
                    emb1.reshape(1, -1), emb2.reshape(1, -1)
                )

                if similarity[0] >= coherence_threshold:
                    # append the tuple with the embedding for each word that's similar
                    coherent_words.append((word1[0], word1[1], emb1))
                    coherent_words.append((word2[0], word2[1], emb2))

        return coherent_words, kw_prev_sentence, kw_curr_sentence

    def get_coherence(self, segment, coherence_threshold: float = 1):
        """creates a list of words that are common and strong in a segment.

        Args:
            segments (list[str]): a segment of sentences to get keywords and collect similar ones on
            coherence_threshold (float): If this number is anything less than one, look for similar words higher than the provided value. Otherwise look for only identical words

        Returns:
            list: list of words that are considered high coherence in the segment
        """
        cohesion = []
        prev_sentence = None
        for sentence in segment:
            if prev_sentence is None:
                prev_sentence = sentence
                continue
            else:
                (
                    coherent_words,
                    kw_prev_sentence,
                    kw_curr_sentence,
                ) = self.get_similar_coherent_words(
                    prev_sentence, sentence, coherence_threshold
                )[
                    : self.max_words_per_step
                ]
                cohesion.extend(coherent_words)
                prev_sentence = sentence

        return cohesion[: self.max_words_per_step], kw_prev_sentence, kw_curr_sentence

    def get_coherence_map(
        self,
        segments,
    ):
        coherence_map = []
        for segment in segments:
            coherence_map.append(self.get_coherence(segment))

        return coherence_map

    def compare_coherent_words(
        self,
        coherence_map,
        keywords_current,
        suppress_errors=True,
    ):
        word_comparisons = []

        # reverse the coherence map and iterate through it so we can go through
        # important words from the closest sentences to the furthest sentences.
        # E.g., s7 -> s6 -> s5 -> s4 -> etc..
        for i, keywords in enumerate(coherence_map[::-1]):
            for word_tuple in keywords:
                word = word_tuple[0]
                for second_word_tuple in keywords_current:
                    second_word = second_word_tuple[0]
                    second_word_importance = second_word_tuple[1]

                    try:
                        word_one_emb = word_tuple[2]
                        word_two_emb = second_word_tuple[2]

                        word_comparisons.append(
                            (
                                word,
                                second_word,
                                self.embedding_lib.get_similarity(
                                    word_one_emb, word_two_emb
                                ),
                            )
                        )
                    except AssertionError as e:
                        if not suppress_errors:
                            print(e, word, second_word)

        return word_comparisons

    def predict(
        self,
        text_data,
        max_tokens=256,
        prediction_threshold=0.25,
        coherence_dump_on_prediction=False,
        pruning=1,  # remove one sentence worth of keywords
        pruning_min=7,  # remove the first sentence in the coherence map once it grows passed 6
        dynamic_threshold=False,
        threshold_warmup=10,  # number of iterations before using dynamic threshold
        last_n_threshold=5,  # will only consider the last n thresholds for dynamic threshold
    ):
        coherence_map = []
        predictions = []
        thresholds = []
        for i, row in enumerate(text_data):
            threshold = prediction_threshold

            # dynamic threshold calculations
            if dynamic_threshold and (i + 1) > threshold_warmup:
                last_n_thresholds = thresholds[(0 - last_n_threshold) :]
                last_n_thresholds.sort()
                mid = len(last_n_thresholds) // 2
                threshold = (last_n_thresholds[mid] + last_n_thresholds[~mid]) / 2
                print(f"median threshold: {threshold}")

            # compare the current sentence to the previous one
            if i == 0:
                predictions.append(
                    (torch.tensor(0, dtype=torch.int8), 0)
                )  # predict a 0 since it's the start
                pass
            else:
                prev_row = text_data[i - 1]

                row = truncate_by_token(row, max_tokens)
                prev_row = truncate_by_token(prev_row, max_tokens)

                # add the keywords to the coherence map
                cohesion, keywords_prev, keywords_current = self.get_coherence(
                    [row, prev_row], coherence_threshold=0.2
                )

                # add the keywords to the coherence map
                coherence_map.append(cohesion)

                # print("coherence map", coherence_map)
                if pruning > 0 and len(coherence_map) >= pruning_min:
                    coherence_map = coherence_map[
                        pruning:
                    ]  # get the last n - pruning values and reverse the list

                # get the keywords for the current sentences
                keywords_current = self.keywords_lib.get_keywords_with_embeddings(row)
                keywords_prev = self.keywords_lib.get_keywords_with_embeddings(prev_row)

                # compute the word comparisons between the previous (with the coherence map)
                # and the current (possibly the first sentence in a new segment)
                weighted_similarities = self.compare_coherent_words(
                    [*coherence_map, keywords_prev], keywords_current
                )

                weighted_similarities = [
                    comparison[2] for comparison in weighted_similarities
                ]
                avg_similarity = sum(weighted_similarities) / len(weighted_similarities)

                # if the two sentences are similar, create a cohesive prediction
                # otherwise, predict a new segment
                if avg_similarity > threshold:
                    predictions.append((avg_similarity, 0))
                else:
                    if coherence_dump_on_prediction:
                        # start of a new segment, empty the map
                        coherence_map = []
                    predictions.append((avg_similarity, 1))

                thresholds.append(avg_similarity)
                print(".", end="")

        return predictions
