from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
count_tokenizer = count_vectorizer.build_tokenizer()

# tokneize a sent


def tokenize(sent):
    tokens = " ".join(word_tokenize(sent.lower()))
    return tokens

# filter some noises caused by speech recognition


def clean_sentence(text):
    text = text.replace("{vocalsound}", "")
    text = text.replace("{disfmarker}", "")
    text = text.replace("a_m_i_", "ami")
    text = text.replace("l_c_d_", "lcd")
    text = text.replace("p_m_s", "pms")
    text = text.replace("t_v_", "tv")
    text = text.replace("{pause}", "")
    text = text.replace("{nonvocalsound}", "")
    text = text.replace("{gap}", "")
    return text


def remove_blank_sentence(arr_text):
    return list(filter(lambda text: len(text) > 0, arr_text))


def preprocess_text_segmentation(data):
    segments = []
    for row in data:
        text = row["text"]
        topics = []
        for annotation in row["annotations"]:
            start = annotation["begin"]
            end = start + annotation["length"]
            topics.append(text[start:end])
        segments.append(topics)

    return segments


def format_data_for_db_insertion(data):
    tuples = []

    id = 1
    # id of the current parent as iterating through
    parent_id = None
    for i, segment in enumerate(data):
        for segment_index, sentence in enumerate(segment):
            # start of the segment
            if segment_index == 0:
                # (autoint id, sentence string, target [1 or 0], parent, sequence)
                current_sentence = (sentence, 1, None, segment_index)
                tuples.append(current_sentence)
                parent_id = id
            # every other sentence in the segment
            else:
                # (autoint id, sentence string, target [1 or 0], parent, sequence)
                current_sentence = (sentence, 0, parent_id, segment_index)
                tuples.append(current_sentence)
            id += 1

    return tuples

# HELPERS


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
