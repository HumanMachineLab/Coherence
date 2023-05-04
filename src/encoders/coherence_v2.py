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
            embedding_technique = self.keywords_lib.get_keywords_with_kb_embeddings
        else:
            embedding_technique = self.keywords_lib.get_keywords_with_embeddings

        kw_curr_sentence = embedding_technique(curr_sentence)[: self.max_words_per_step]
        kw_prev_sentence = embedding_technique(prev_sentence)[: self.max_words_per_step]

        coherent_words = []

        for word2 in kw_curr_sentence:
            for word1 in kw_prev_sentence:
                emb1 = torch.Tensor(word1[2])
                emb2 = torch.Tensor(word2[2])

                # check to see if either word by its embedding already exists in the
                # coherent words so far.
                skip_comparison = False
                coherent_word_embeddings_only = [w[2] for w in coherent_words]
                for we in coherent_word_embeddings_only:
                    if torch.equal(we, emb1) or torch.equal(we, emb2):
                        # the word has already been added
                        skip_comparison = True
                        continue

                # # don't consider all numbers because in a pre-trained LLM
                # # they have no use or meaning.
                # if word1[0].isnumeric() or word2[0].isnumeric():
                #     skip_comparison = True
                #     continue

                if not skip_comparison:
                    # check similarity and add to coherent dictionary
                    similarity = torch.cosine_similarity(
                        emb1.reshape(1, -1), emb2.reshape(1, -1)
                    )

                    if similarity[0] >= coherence_threshold:
                        # append the tuple with the embedding for each word that's similar
                        coherent_words.append((word1[0], word1[1], emb1))
                        coherent_words.append((word2[0], word2[1], emb2))

        # sort by descending to have the most important words first
        desc_sorted_words = sorted(coherent_words, key=lambda x: x[1])[::-1]
        return desc_sorted_words, kw_prev_sentence, kw_curr_sentence

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

    # get the weighted average of keywords collected in the coherence map thus far
    def get_weighted_average(self, weighted_similarities, weights):
        return sum(weighted_similarities) / sum(weights)

    def compare_coherent_words(
        self,
        coherence_map,
        keywords_current,
        suppress_errors=True,
    ):
        word_comparisons = []
        weights = []

        # reverse the coherence map and iterate through it so we can go through
        # important words from the closest sentences to the furthest sentences.
        # E.g., s7 -> s6 -> s5 -> s4 -> etc..
        for i, keywords in enumerate(coherence_map[::-1]):
            for word_tuple in keywords:
                word = word_tuple[0]
                for second_word_tuple in keywords_current:
                    second_word = second_word_tuple[0]
                    second_word_importance = second_word_tuple[1]

                    # this is the value that is used to identify the strength of the
                    # word in relation to its sentence as provided by keybert originally
                    second_word_importance = second_word_tuple[1]

                    try:
                        word_one_emb = word_tuple[2]
                        word_two_emb = second_word_tuple[2]

                        if self.same_word_multiplier > 1:
                            flattened_coherence_words_only = [
                                element[0]
                                for sublist in coherence_map
                                for element in sublist
                            ]

                            # if the current word shows up a lot in the coherence map
                            # we want to amplify it by the provided amplifier value
                            num_occurrences = flattened_coherence_words_only.count(
                                second_word
                            )

                            if num_occurrences > 0:
                                # amplify words that are found as duplicates in the coherence map
                                # if the word shows up 1 time, amplify the weight by 2 times
                                weighting_multiplier = (
                                    flattened_coherence_words_only.count(second_word)
                                    + (self.same_word_multiplier - 1)
                                )
                            else:
                                # no same word penalty
                                weighting_multiplier = (
                                    1 / self.no_same_word_penalty
                                )  # reduce the importance of this word

                        else:
                            weighting_multiplier = (
                                1  # set to 1 in case this is turned off.
                            )

                        # this weight is a recipricol function that will grow smaller the further the keywords are away
                        # we want to put more importance on the current words, so we apply twice as much weight.
                        if i == 0:
                            weight = (weighting_multiplier * 2) / (i + 1)
                        else:
                            weight = (weighting_multiplier * 1) / (i + 1)

                        # multiply the weighting factor by the importance of the second word
                        weight *= second_word_importance

                        # calculate the comparison between the current word and the word being
                        # iterated over in the coherence map
                        # word_weight (from KB) * weighting_multiplier * similarity
                        word_comparison = weight * self.embedding_lib.get_similarity(
                            word_one_emb, word_two_emb
                        )

                        word_comparisons.append((word, second_word, word_comparison))

                        # We need to get the weights and store them for the weighted average later
                        weights.append(weight)
                    except AssertionError as e:
                        if not suppress_errors:
                            print(e, word, second_word)

        return word_comparisons, weights

    def predict(
        self,
        text_data,
        max_tokens=128,
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

                # compute the word comparisons between the previous (with the coherence map)
                # and the current (possibly the first sentence in a new segment)
                weighted_similarities, weights = self.compare_coherent_words(
                    [*coherence_map, keywords_prev], keywords_current
                )

                weighted_similarities_values_only = [
                    comparison[2] for comparison in weighted_similarities
                ]

                # get the average weighted similarity as calculated from above
                avg_similarity = self.get_weighted_average(
                    weighted_similarities_values_only, weights
                )

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
