import argparse
import copy
import json
import logging
import pickle
import traceback

import pandas as pd
import numpy as np

from rule import AnsFunc, Variable, Rule

from parser import Parser
from parser import preprocess_sent, tokenize

from utils.squad_reader import SquadReader
from utils.squad_utils import exact_match
from utils.general_utils import get_logger

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--exp_file", default='./explanations/squad.csv',
                    type=str, required=False)
arg_parser.add_argument("--ans_func_file", default='./explanations/squad.pkl',
                    type=str, required=False)
arg_parser.add_argument('--save_ans_func', action='store_true',
                    help='If true, will save a pkl file to --ans_func_file')
arg_parser.add_argument('--unit_test', action='store_true',
                    help='If true, parse the first row in csv file.')
arg_parser.add_argument('--verbose', action='store_true',
                    help='If true, print every hard-matched instance')

args = arg_parser.parse_args()
opt = vars(args)

logger = get_logger(__name__, stream_handler=False)

def get_ans_func(row, parser, squad_reader):
    question, context, answer, _, X, Y, Z, Exp = row
    context = context.replace("``", "\"").replace("''", "\"").replace("<font color=\\\"red\\\"> ", "").replace(" </font>", "").replace("<font color=\"red\">", "").replace("</font>", "")
    answer = answer.strip('!\'*+,-./:;<=>?@[\\]^_`{|}~')
    squad_instance = squad_reader.read_one(context, question, answer)

    ansFunc = AnsFunc()
    ansFunc.instantiate(squad_instance)
    ansFunc.add_variable(Variable.get_variable(squad_instance, answer, 'Answer'))

    Exp = Exp.strip(' ')
    if len(X) > 0 and (X.lower() in question.lower() or X.lower() in context.lower()):
        ansFunc.add_variable(Variable.get_variable(squad_instance, X, 'X'))
        Exp += ' X appears in the context.'
    if len(Y) > 0 and (Y.lower() in question.lower() or Y.lower() in context.lower()):
        ansFunc.add_variable(Variable.get_variable(squad_instance, Y, 'Y'))
        Exp += ' Y appears in the context.'
    if len(Z) > 0 and (Z.lower() in question.lower() or Z.lower() in context.lower()):
        ansFunc.add_variable(Variable.get_variable(squad_instance, Z, 'Z'))
        Exp += ' Z appears in the context.'


    inputs = {'Answer': {'span': answer,
                         'begin_idx': squad_instance.start_position[0],
                         'end_idx': squad_instance.end_position[0],
                         'sentence_idx': squad_instance.idx_sentence_containing_answer[0]},
              'X': X,
              'Y': Y,
              'Z': Z,
              'all': question + ' ' + context,
              'instance': squad_instance,
              'pretrained_modules': None
              }

    sentences = preprocess_sent(Exp)
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    for sentence in tokenized_sentences:
        rule_list_sentence = []
        for potential_sentence in sentence:
            all_possible_parses = parser.parse(potential_sentence)
            if len(all_possible_parses) > 0:
                rule_list_sentence += all_possible_parses
        if len(rule_list_sentence) > 0:
            for a_rule in rule_list_sentence:
                rule = Rule(a_rule)
                inputs_copy = copy.copy(inputs)
                if np.equal(1, rule.execute(inputs_copy)):
                    ansFunc.add_rule(a_rule)
                    break

    # ans_by_ansFunc, (st, ed) = get_majority(ansFunc.answer(squad_instance))
    logging.info([item.raw_str for item in ansFunc.all_rules])
    all_answers = ansFunc.answer(squad_instance, pretrained_modules=None, soft=False, thres=1.0)

    validated = False
    answer_tokenized = ' '.join([item.text for item in squad_instance.answer_tokens[0]])

    if len(all_answers) == 1:
        one_answer = all_answers[0]
        validated = exact_match(one_answer['Answer'].span, answer_tokenized)

    return ansFunc, validated


def read_csv(filename):
    df = pd.read_csv(filename).fillna("")
    df = df[['Input.question', 'Input.context', 'Input.answer', 'Input.qas_id',
             'Answer.X', 'Answer.Y', 'Answer.Z', 'Answer.Exp']]
    return df

def main():

    df = read_csv(opt["exp_file"])
    parser = Parser()

    ans_funcs = []
    valid_idx = []

    squad_reader = SquadReader(pre=True)

    for idx, row in enumerate(df.iterrows()):
        try:
            logger.info('Processing Rule #{}'.format(idx))
            ansFunc, validated = get_ans_func(df.iloc[idx], parser, squad_reader)
            logger.info('Rule #{} Validated? {}'.format(idx, str(validated)))

            if opt['verbose']:
                logger.info('=======Rule #{}======='.format(idx))
                logger.info(str([item.raw_str for item in ansFunc.all_rules]))

            if validated:
                ans_funcs.append((idx, ansFunc))
                valid_idx.append(idx)

            if opt['unit_test'] and idx == 0:
                break

        except Exception as e:
            logger.info(traceback.format_exc())
            logger.info('Error when validating rule #{}'.format(idx))
            logger.info('Error information: {}'.format(e.args))

    logger.info('#All: {}, #Validated: {}, Validated idx: {}'.format(len(df), len(ans_funcs), valid_idx))

    if opt['save_ans_func']:
        for idx, ans_func in ans_funcs:
            ans_func.delete_redundant_rules()
            ans_func.clean_rules() # lambda functions cannot be binarized.
        pickle.dump(ans_funcs, open(opt['ans_func_file'], 'wb'))
        logger.info('Dumped {} ans funcs to {}'.format(len(ans_funcs), opt['ans_func_file']))

if __name__ == "__main__":
    main()