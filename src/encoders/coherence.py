import torch
from src.bertkeywords.src.similarities import Embedding, Similarities
from src.bertkeywords.src.keywords import Keywords
from src.dataset.utils import dedupe_list, flatten, truncate_string, truncate_by_token
import time


class Coherence:
    def __init__(self, max_words_per_step=2, coherence_threshold=0.4):
        self.max_words_per_step = max_words_per_step
        self.coherence_threshold = coherence_threshold
        # similarities_lib = Similarities("sentence-transformers/LaBSE")
        similarities_lib = Similarities("bert-base-uncased")
        # similarities_lib = Similarities("roberta-base")
        # similarities_lib = Similarities("sentence-transformers/all-MiniLM-L6-v2")
        # similarities_lib = Similarities("Dimitre/universal-sentence-encoder")

        self.keywords_lib = Keywords(similarities_lib.model, similarities_lib.tokenizer)
        self.embedding_lib = Embedding(
            similarities_lib.model, similarities_lib.tokenizer
        )

    def get_similar_coherent_words(
        self, prev_sentence, curr_sentence, coherence_threshold
    ):
        tic = time.perf_counter()
        kw_curr_sentence = self.keywords_lib.get_keywords_with_embeddings(
            curr_sentence
        )[: self.max_words_per_step]
        kw_prev_sentence = self.keywords_lib.get_keywords_with_embeddings(
            prev_sentence
        )[: self.max_words_per_step]
        print([x[0] for x in kw_curr_sentence], curr_sentence)
        print([x[0] for x in kw_prev_sentence], prev_sentence)
        toc = time.perf_counter()
        print(f"Got the keywords in {toc - tic:0.4f} seconds")

        coherent_words = []

        tic = time.perf_counter()
        for word2 in kw_curr_sentence:
            for word1 in kw_prev_sentence:
                # check to see if either word by its embedding already exists in the
                # coherent words so far.
                skip_comparison = False
                coherent_word_embeddings_only = [w[2] for w in coherent_words]
                for we in coherent_word_embeddings_only:
                    if torch.equal(we, word1[2]) or torch.equal(we, word2[2]):
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
                    emb1 = word1[2]
                    emb2 = word2[2]
                    similarity = torch.cosine_similarity(
                        emb1.reshape(1, -1), emb2.reshape(1, -1)
                    )

                    if similarity[0] >= coherence_threshold:
                        # append the tuple with the embedding for each word that's similar
                        coherent_words.append((word1[0], word1[1], emb1))
                        coherent_words.append((word2[0], word2[1], emb2))

        toc = time.perf_counter()
        print(f"Got the embeddings and comparisons in {toc - tic:0.4f} seconds")

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
        same_word_multiplier=True,
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

                    try:
                        word_one_emb = word_tuple[2]
                        word_two_emb = second_word_tuple[2]

                        if same_word_multiplier:
                            flattened_coherence_words_only = [
                                element[0]
                                for sublist in coherence_map
                                for element in sublist
                            ]

                            num_occurrences = flattened_coherence_words_only.count(
                                second_word
                            )

                            multiplier = 2
                            if num_occurrences > 0:
                                # amplify words that are found as duplicates in the coherence map
                                # if the word shows up 1 time, amplify the weight by 2 times
                                weighting_multiplier = (
                                    flattened_coherence_words_only.count(second_word)
                                    + (multiplier - 1)
                                )
                            else:
                                weighting_multiplier = (
                                    1 / multiplier
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

                        word_comparisons.append(
                            (
                                word,
                                second_word,
                                weight
                                * self.embedding_lib.get_similarity(
                                    word_one_emb, word_two_emb
                                ),
                            )
                        )
                        weights.append(weight)
                    except AssertionError as e:
                        if not suppress_errors:
                            print(e, word, second_word)

    def predict(
        self,
        text_data,
        max_tokens=256,
        prediction_thresh=0.3,
        pruning=1,
        pruning_min=4,
        threshold_warmup=10,  # number of iterations before using dynamic threshold
        last_n_threshold=5,  # will only consider the last n thresholds for dynamic threshold
        dynamic_threshold=False,
    ):
        coherence_map = []
        predictions = []
        thresholds = []
        for i, row in enumerate(text_data):
            threshold = prediction_thresh

            # dynamic threshold calculations
            if dynamic_threshold and (i + 1) > threshold_warmup:
                last_n_thresholds = thresholds[(0 - last_n_threshold) :]
                last_n_thresholds.sort()
                mid = len(last_n_thresholds) // 2
                threshold = (last_n_thresholds[mid] + last_n_thresholds[~mid]) / 2
                print(f"median threshold: {threshold}")

            # compare the current sentence to the previous one
            if i == 0:
                predictions.append((0, 0))  # predict a 0 since it's the start
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
                if pruning > 0 and len(coherence_map) >= pruning_min:
                    coherence_map = coherence_map[::-1][pruning:][
                        ::-1
                    ]  # get the last n - pruning values and reverse the list

                # get the keywords for the current sentences
                keywords_current = self.keywords_lib.get_keywords_with_embeddings(row)
                keywords_prev = self.keywords_lib.get_keywords_with_embeddings(prev_row)

                # compute the word comparisons between the previous (with the coherence map)
                # and the current (possibly the first sentence in a new segment)
                weighted_similarities, weights = self.compare_coherent_words(
                    [*coherence_map, keywords_prev], keywords_current
                )

                weighted_similarities = [
                    comparison[2] for comparison in weighted_similarities
                ]
                avg_similarity = self.get_weighted_average(
                    weighted_similarities, weights
                )

                # if the two sentences are similar, create a cohesive prediction
                # otherwise, predict a new segment
                if avg_similarity > threshold:
                    predictions.append((avg_similarity, 0))
                else:
                    # start of a new segment, empty the map
                    coherence_map = []
                    predictions.append((avg_similarity, 1))

                thresholds.append(avg_similarity)
                print(".", end="")

        return predictions

    # testing functions
    def predict_verbose(
        self,
        text_data,
        max_tokens=256,
        prediction_thresh=0.3,
        pruning=4,
        pruning_min=10,
    ):
        coherence_map = []
        predictions = []
        for i, row in enumerate(text_data):
            # compare the current sentence to the previous one
            if i == 0:
                predictions.append((0, 0))  # predict a 0 since it's the start
                pass
            else:
                prev_row = text_data[i - 1]

                row = truncate_by_token(row, max_tokens)
                prev_row = truncate_by_token(prev_row, max_tokens)

                # add the keywords to the coherence map
                coherence_map.extend(
                    self.get_coherence(
                        [row, prev_row], coherence_threshold=self.coherence_threshold
                    )
                )
                print(f"Coherence Map: {[[x[0] for x in c] for c in coherence_map]}")
                if pruning > 0 and len(coherence_map) >= pruning_min:
                    sorted_map = sorted(
                        coherence_map, key=lambda tup: tup[1]
                    )  # sort asc by importance based on keybert
                    coherence_map = sorted_map[pruning:][
                        ::-1
                    ]  # get the last n - pruning values and reverse the list

                # get the keywords for the current sentences
                keywords_current = self.keywords_lib.get_keywords_with_embeddings(row)
                keywords_prev = self.keywords_lib.get_keywords_with_embeddings(prev_row)
                print(
                    f"Coherence Map: {[[x[0] for x in c] for c in coherence_map]}, Keywords Current: {[x[0] for x in keywords_current]}"
                )

                # compute the word comparisons between the previous (with the coherence map)
                # and the current (possibly the first sentence in a new segment)
                weighted_similarities, weights = self.compare_coherent_words(
                    [*coherence_map, keywords_prev], keywords_current
                )

                weighted_similarities = [
                    comparison[2] for comparison in weighted_similarities
                ]
                avg_similarity = self.get_weighted_average(
                    weighted_similarities, weights
                )

                # if the two sentences are similar, create a cohesive prediction
                # otherwise, predict a new segment
                if avg_similarity > prediction_thresh:
                    print(f"Similarity: {avg_similarity}, Prediction: {0}")
                    predictions.append((avg_similarity, 0))
                else:
                    # start of a new segment, empty the map
                    coherence_map = []
                    print(f"Similarity: {avg_similarity}, Prediction: {1}")
                    predictions.append((avg_similarity, 1))

        return predictions
