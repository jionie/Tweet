def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def jaccard_v2(list1, list2):
    a = set(list1)
    b = set(list2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def calculate_jaccard_score(
        original_tweet,
        selected_text,
        idx_start,
        idx_end,
        model_type,
        tweet_offsets):

    if idx_end < idx_start:
        length = len(original_tweet)
        filtered_output = original_tweet[length // 4 : length // 4 * 3]
        # filtered_output = original_tweet

    else:

        if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":

            # we remove first tokens in hidden states
            idx_start += 4
            idx_end += 4

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            # print(selected_text, "----------", filtered_output)

        elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (
                model_type == "albert-xlarge-v2"):

            # we remove first tokens in hidden states
            idx_start += 3
            idx_end += 3

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            # print(selected_text, "----------", filtered_output)

        elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):

            # we remove first tokens in hidden states
            idx_start += 2
            idx_end += 2

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            # print(selected_text, filtered_output)

        elif (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):

            # we remove first tokens in hidden states
            idx_start += 3
            idx_end += 3

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            # print(selected_text, filtered_output)

        elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

            # we remove first tokens in hidden states
            idx_start += 3
            idx_end += 3

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            # print(selected_text, filtered_output)

        else:
            raise NotImplementedError

    jac = jaccard(selected_text.strip(), filtered_output.strip())
    return jac, filtered_output