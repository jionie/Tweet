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


def calculate_spaces(text, selected_text):
    index = text.index(selected_text)
    x = text[:index]
    try:
        if x[-1]==" ":
            x= x[:-1]
    except:
        pass
    l1 = len(x)
    l2 = len(" ".join(x.split()))
    return l1-l2


def find_all(word, text):
    import re
    word = word.replace(".","\.")
    word = word.replace(")","\)")
    word = word.replace("(","\(")
    word = word.replace("?","\?")
    word = word.replace("!","\!")
    word = word.replace("*","\*")
    word = word.replace("$","\$")
    word = word.replace("[","\[")
    word = word.replace("]","\]")
    word = word.replace("+","\+")
    word = word.replace(",","\,")
    return [m.start() for m in re.finditer(word, text)]


def extract_end_index(text, selected_text):
    i=0
    last_word = selected_text.split()[-1]
    index_last_word = find_all(last_word, text)
    n_occ = len(index_last_word)
    selected_text_split = selected_text.split()
    text_split = text.split()
    n_end = 0
    if len(selected_text_split)==len(text_split):
        return len(text)
    for j, elm in enumerate(text_split[len(selected_text_split):]):
        i = j + len(selected_text_split)
        if elm == last_word :
            n_end +=1
            if text_split[j+1:i+1] == selected_text_split:
                break
    return index_last_word[n_end-1] + len(selected_text_split[-1])


def extract_start_index(text, selected_text):
    first_word = selected_text.split()[0]
    index_first_word = find_all(first_word, text)
    selected_text_split = selected_text.split()
    text_split = text.split()
    n_start = 0
    for i, elm in enumerate(text_split):
        if (first_word !=elm) and (first_word in elm):
            n_start += elm.count(first_word)
        if elm == first_word :
            n_start +=1
            if text_split[i:i+len(selected_text_split)] == selected_text_split:
                break
    return index_first_word[n_start-1]


def pp_v2(text, predicted, spaces):
    text = text.lower()
    predicted = predicted.lower()
    predicted = predicted.strip()
    index_start = extract_start_index(text, predicted)
    index_end = extract_end_index(text, predicted)
    if spaces == 1:
        if len(text[max(0, index_start-1): index_end+1]) <= 0 or text[max(0, index_start-1): index_end+1][-1] != ".":
            return text[max(0, index_start - 1): index_end]
        else:
            return text[max(0, index_start-1): index_end+1]
    elif spaces == 2:
        return text[max(0, index_start-2): index_end]
    elif spaces == 3:
        return text[max(0, index_start-3): index_end-1]
    elif spaces == 4:
        return text[max(0, index_start-4): index_end-2]
    else:
        return predicted


def calculate_jaccard_score(
        original_tweet,
        target_string,
        idx_start,
        idx_end,
        sentiment,
        tokenizer):

    if idx_end < idx_start:
        length = len(original_tweet)
        filtered_output = original_tweet
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
        input_ids = input_ids_orig + [2]
        filtered_output = tokenizer.decode(input_ids[idx_start:idx_end+1])

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output