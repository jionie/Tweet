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
        sentiment,
        tokenizer):

    if idx_end < idx_start:
        length = len(original_tweet)
        filtered_output = original_tweet[length // 4 : length // 4 * 3]
    else:
        # filtered_output = ""
        # for ix in range(idx_start, idx_end + 1):
        #     filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        #     if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
        #         filtered_output += " "
        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }
        input_ids_orig = tokenizer.encode(original_tweet).ids
        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        filtered_output = tokenizer.decode(input_ids[idx_start:idx_end+1])

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output