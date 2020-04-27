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
        tokenizer):

    if idx_end < idx_start:
        length = len(original_tweet)
        # filtered_output = original_tweet[length // 4 : length // 4 * 3]
        filtered_output = original_tweet
    else:
        input_ids_orig = tokenizer.encode(original_tweet).ids
        input_ids = input_ids_orig + [2]
        filtered_output = tokenizer.decode(input_ids[idx_start:idx_end+1])

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output