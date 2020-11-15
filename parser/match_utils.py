import json
import string
from collections import Counter

def get_st_ed(final_answer, answers):
    """Given an answer string and a list of potential answers, backtrace its location."""
    st, ed, max_confidence = -1, -1, 0.0
    for answer in answers:
        value = answer['Answer'].span
        if isinstance(value, list):
            value = ' '.join(value)
        if value.strip(string.punctuation).strip() == final_answer:
            offset = answer['Answer'].offset
            st = answer['Answer'].begin_idx + offset
            ed = answer['Answer'].end_idx + offset
            confidence = answer['confidence']
            max_confidence = max(max_confidence, confidence)
    return (st, ed, max_confidence)

def get_majority_hard(answers):
    """if multiple answers are selected by different rules, we perform majority vote.
    if the top 2 answers have the same votes, we abstain.
    
    Return: answer string, (start_idx, end_idx, confidence)
    """

    if len(answers) == 0:
        return '', (-1, -1, 0.0)

    cnt = Counter()
    for ans in answers:
        value = ans['Answer'].span
        if isinstance(value, list):
            value = ' '.join(value)
        cnt[value.strip(string.punctuation).strip()] += 1

    if len(cnt) == 1: # only one answer appear
        ans1, count1 = cnt.most_common(1)[0]
        return ans1, get_st_ed(ans1, answers)
    else: # compare top 2 answers
        (ans1, count1) , (ans2, count2) = cnt.most_common(2)

        # if the top 2 answers give a draw
        if count1 == count2:
            # and if answer1 is in the answer 2, we prefer the longer answer (answer1)
            if ans2 in ans1:
                return ans1, get_st_ed(ans1, answers)
            elif ans1 in ans2:
                return ans2, get_st_ed(ans2, answers)
            # if it's still a draw, we abstain
            else:
                return '', (-1, -1, 0.0)
        else: # not a draw, ans1 gets more votes
            return ans1, get_st_ed(ans1, answers)


def save_matched(instances, filename):
    instances_to_save = []
    for qa_instance, answer, (st, ed, confidence) in instances:
        doc_tokens = []
        for sentence in qa_instance.context_info:
            doc_tokens += sentence['tokens']
        new_instance = {
            "qas_id": qa_instance.qas_id,
            "question_text": qa_instance.question_text,
            "doc_tokens": doc_tokens,
            "orig_answer_text": answer,
            "start_position": st,
            "end_position": ed,
            "is_impossible": 0,
            "weight": confidence
        }
        instances_to_save.append(new_instance)

    with open(filename, 'w') as fout:
        json.dump(instances_to_save, fout)
