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
        target_string,
        idx_start,
        idx_end,
        model_type,
        tokenizer):

    if idx_end < idx_start:
        length = len(original_tweet)
        filtered_output = original_tweet[length // 4 : length // 4 * 3]
        # filtered_output = original_tweet
    else:
        input_ids_orig = tokenizer.encode(original_tweet)

        if (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
            # sep and cls will added
            input_ids_orig = input_ids_orig[:-2]

        elif (model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad"):
            input_ids_orig = input_ids_orig

        else:
            # cls and sep will added
            input_ids_orig = input_ids_orig[1:-1]

        tweet_offsets = []
        idx = 0
        for t in input_ids_orig:
            w = tokenizer.decode([t])
            tweet_offsets.append((idx, idx + len(w)))
            idx += len(w)

        if idx_start >= len(tweet_offsets):
            idx_start = 0
        if idx_end >= len(tweet_offsets):
            idx_end = len(tweet_offsets) - 1

        if (model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad"):
            filtered_output = ""
            for ix in range(idx_start, idx_end + 1):
                filtered_output += original_tweet[tweet_offsets[ix][0]: tweet_offsets[ix][1]]
                if (ix + 1) < len(tweet_offsets) and tweet_offsets[ix][1] < tweet_offsets[ix + 1][0]:
                    filtered_output += " "
        else:
            tweet_copy = "".join(original_tweet.split())
            prediction = ""
            orig_idx0 = 0
            filtered_output = original_tweet
            for ix in range(idx_start, idx_end + 1):
                prediction += tweet_copy[tweet_offsets[ix][0]: tweet_offsets[ix][1]]
                if (ix + 1) < len(tweet_offsets) and tweet_offsets[ix][1] < tweet_offsets[ix + 1][0]:
                    prediction += " "

            if prediction == "":
                prediction = "".join(original_tweet.split())
            len_prediction = len(prediction)

            for ind in (i for i, e in enumerate(original_tweet) if e == prediction[0]):

                tweet_sub_sentence = "".join(original_tweet[ind:].split())

                if tweet_sub_sentence[:len_prediction] == prediction:
                    orig_idx0 = ind
                    break

            for end_ind in range(orig_idx0, len(original_tweet)):
                if "".join(original_tweet[orig_idx0: end_ind+1].split()) == prediction:
                    filtered_output = original_tweet[orig_idx0: end_ind+1]
                    break
            # print(prediction, filtered_output)

        # input_ids = input_ids_orig
        # filtered_output = tokenizer.decode(input_ids[idx_start:idx_end+1])

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output