import numpy as np

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

def get_word_level_logits(start_logits,
                          end_logits,
                          model_type,
                          tweet_offsets_word_level):

    tweet_offsets_word_level = np.array(tweet_offsets_word_level)

    if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":
        logit_offset = 4

    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (
            model_type == "albert-xlarge-v2"):
        logit_offset = 3

    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        logit_offset = 2

    elif (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):
        logit_offset = 3

    elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):
        logit_offset = 3

    prev = tweet_offsets_word_level[logit_offset]
    word_level_bbx = []
    curr_bbx = []

    for i in range(len(tweet_offsets_word_level) - logit_offset - 1):

        curr = tweet_offsets_word_level[i + logit_offset]

        if curr[0] < prev[0] and curr[1] > prev[1]:
            break

        if curr[0] == prev[0] and curr[1] == prev[1]:
            curr_bbx.append(i)
        else:
            word_level_bbx.append(curr_bbx)
            curr_bbx = [i]

        prev = curr

    if len(word_level_bbx) == 0:
        word_level_bbx.append(curr_bbx)

    # debug word_level_bbx
    # tmp_bbx = []
    # for i in range(len(word_level_bbx)):
    #     tmp = []
    #
    #     for bbx in word_level_bbx[i]:
    #         tmp.append(tuple(tweet_offsets_word_level[bbx+logit_offset]))
    #
    #     tmp_bbx.append(tmp)
    #
    # print(tmp_bbx)

    for i in range(len(word_level_bbx)):
        word_level_bbx[i].append(word_level_bbx[i][-1] + 1)

    start_logits_word_level = [np.max(start_logits[bbx[0]: bbx[-1]]) for bbx in word_level_bbx]
    end_logits_word_level = [np.max(end_logits[bbx[0]: bbx[-1]]) for bbx in word_level_bbx]

    return start_logits_word_level, end_logits_word_level, word_level_bbx


def get_token_level_idx(start_logits,
                        end_logits,
                        start_logits_word_level,
                        end_logits_word_level,
                        word_level_bbx):

    # get most possible word
    start_idx_word = np.argmax(start_logits_word_level)
    end_idx_word = np.argmax(end_logits_word_level)

    # get all token idx in selected word
    start_word_bbx = word_level_bbx[start_idx_word]
    end_word_bbx = word_level_bbx[end_idx_word]

    # find most possible token idx in selected word
    start_idx_in_word = np.argmax(start_logits[start_word_bbx[0]: start_word_bbx[-1]])
    end_idx_in_word = np.argmax(end_logits[end_word_bbx[0]: end_word_bbx[-1]])

    # find most possible token idx in whole sentence
    start_idx_token = start_word_bbx[start_idx_in_word]
    end_idx_token = end_word_bbx[end_idx_in_word]

    return start_idx_token, end_idx_token
    # return np.argmax(start_logits), np.argmax(end_logits)


def get_selected_text(text, start_idx, end_idx, offsets):

    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def calculate_jaccard_score(
        original_tweet,
        selected_text,
        idx_start,
        idx_end,
        start_gt,
        end_gt,
        model_type,
        tweet_offsets):

    if idx_end < idx_start:
        filtered_output = original_tweet

    else:

        if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":

            # we remove first tokens in hidden states
            idx_start += 4
            idx_end += 4

            # filtered_output = get_selected_text(original_tweet, idx_start, idx_end, tweet_offsets)
            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            selected_text = original_tweet[tweet_offsets[start_gt][0]: tweet_offsets[end_gt][1]]
            # print(selected_text, "----------", filtered_output)

        elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (
                model_type == "albert-xlarge-v2"):

            # we remove first tokens in hidden states
            idx_start += 3
            idx_end += 3

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            selected_text = original_tweet[tweet_offsets[start_gt][0]: tweet_offsets[end_gt][1]]
            # print(selected_text, "----------", filtered_output)

        elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):

            # we remove first tokens in hidden states
            idx_start += 2
            idx_end += 2

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            selected_text = original_tweet[tweet_offsets[start_gt][0]: tweet_offsets[end_gt][1]]
            # print(selected_text, filtered_output)

        elif (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):

            # we remove first tokens in hidden states
            idx_start += 3
            idx_end += 3

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            selected_text = original_tweet[tweet_offsets[start_gt][0]: tweet_offsets[end_gt][1]]
            # print(selected_text, filtered_output)

        elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

            # we remove first tokens in hidden states
            idx_start += 3
            idx_end += 3

            filtered_output = original_tweet[tweet_offsets[idx_start][0]: tweet_offsets[idx_end][1]]
            selected_text = original_tweet[tweet_offsets[start_gt][0]: tweet_offsets[end_gt][1]]
            # print(selected_text, filtered_output)

        else:
            raise NotImplementedError

    common_words = ['****.']

    for word in common_words:
        if word in str(original_tweet).lower().split() and word not in str(filtered_output).lower().split():
            filtered_output += " " + word

    jac = jaccard(selected_text.strip(), filtered_output.strip())
    return jac, filtered_output