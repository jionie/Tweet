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
        tweet,
        target_string,
        idx_start,
        idx_end,
        tokenizer,
        offsets):

    if idx_end < idx_start:
        filtered_output = ""
    else:
        # filtered_output = ""
        # for ix in range(idx_start, idx_end + 1):
        #     filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        #     if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
        #         filtered_output += " "
        input_ids_orig = tokenizer.encode(tweet).ids
        input_ids = input_ids_orig + [2]
        selected_tokens = input_ids[idx_start:idx_end+1]
        remove_end_idx = len(selected_tokens) - 1

        while (remove_end_idx >= 0) and (selected_tokens[remove_end_idx] == 2 or selected_tokens[remove_end_idx] == 1):
            remove_end_idx -= 1

        if remove_end_idx < 0:
            length = len(original_tweet)
            filtered_output = original_tweet[length // 4: length // 4 * 3]
        else:

            prediction = tokenizer.decode(selected_tokens[:remove_end_idx+1]).strip()
            original_tweet = original_tweet.lower().strip()

            filtered_output = ""
            finished = False

            # print("tweet: ", tweet)
            # print("prediction: ", prediction)

            if len(prediction) == 0:
                filtered_output = prediction
            else:
                for idx in range(len(original_tweet)):
                    if original_tweet[idx] == prediction[0]:
                        tweet_idx = idx
                        for prediction_idx in range(len(prediction)):
                            if (original_tweet[tweet_idx] == prediction[prediction_idx]):
                                filtered_output += original_tweet[tweet_idx]
                                tweet_idx += 1

                                # reach tweet end
                                if (tweet_idx >= len(original_tweet)):
                                    if (prediction_idx == len(prediction) - 1):
                                        finished = True
                                    break

                                # reach prediction end
                                if (prediction_idx == len(prediction) - 1):
                                    finished = True
                                    break
                            else:
                                # skip extra " "
                                if (prediction[prediction_idx] == " "):
                                    continue
                                else:
                                    filtered_output = ""
                                    break
                        if finished:
                            break

            if len(prediction) != 0 and len(filtered_output) == 0:
                print("orig_tweet: ", original_tweet)
                print("prediction: ", prediction)
                print("output: ", filtered_output)

    if filtered_output == "":
        length = len(original_tweet)
        filtered_output = original_tweet[length // 4: length // 4 * 3]

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output