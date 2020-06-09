import re


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


def find_text_idx(text, selected_text):

    text_len = len(text)

    for start_idx in range(text_len):
        if text[start_idx] == selected_text[0]:
            for end_idx in range(start_idx+1, text_len+1):
                contained_text = " ".join(text[start_idx: end_idx].split())
                # print("contained_text:", contained_text, "selected_text:", selected_text)
                if contained_text == selected_text:
                    return start_idx, end_idx
                if len(contained_text) > len(selected_text):
                    break

    return None, None


def calculate_spaces(text, selected_text):

    selected_text = " ".join(selected_text.split())
    start_idx, end_idx = find_text_idx(text, selected_text)
    # print("text:", text[start_idx: end_idx], "prediction:", selected_text)

    if start_idx is None:
        start_idx = 0
        print("----------------- error no start idx find ------------------")
        print("text:", text, "prediction:", selected_text)
        print("----------------- error no start idx find ------------------")

    if end_idx is None:
        end_idx = len(text)
        print("----------------- error no end idx find ------------------")
        print("text:", text, "prediction:", selected_text)
        print("----------------- error no end idx find ------------------")

    x = text[:start_idx]
    try:
        if x[-1] == " ":
            x = x[:-1]
    except:
        pass

    l1 = len(x)
    l2 = len(" ".join(x.split()))
    return l1 - l2, start_idx, end_idx


def pp_v2(text, predicted):

    text = text.lower()
    predicted = predicted.lower()
    predicted = predicted.strip()
    spaces, index_start, index_end = calculate_spaces(text, predicted)

    if len(predicted.split()) == 1:
        predicted.replace('!!!!', '!')
    if len(predicted.split()) == 1:
        predicted.replace('..', '.')
    if len(predicted.split()) == 1:
        predicted.replace('...', '.')

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
        tokenizer):

    if idx_end < idx_start:
        filtered_output = original_tweet
    else:
        input_ids_orig = tokenizer.encode(original_tweet).ids
        input_ids = input_ids_orig

        if idx_start > len(input_ids):
            idx_start = 0
        if idx_end + 1 > len(input_ids):
            idx_end = len(input_ids) - 1

        filtered_output = tokenizer.decode(input_ids[idx_start: idx_end+1])

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output