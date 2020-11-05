import copy
import numpy as np

from rule import AnsFunc, Rule, Variable
from parser import Parser
from parser import preprocess_sent, tokenize

from utils.squad_reader import SquadReader
from utils.squad_utils import exact_match

def main():

    squad_reader = SquadReader(pre=True)
    parser = Parser()

    question = "How long has Switzerland traditionally been neutral?"
    context = "Traditionally, Switzerland avoids alliances that might entail military, political, or direct economic action and has been neutral <font color=\"red\">since the end of its expansion in 1515.</font> Its policy of neutrality was internationally recognised at the Congress of Vienna in 1815. Only in 2002 did Switzerland become a full member of the United Nations and it was the first state to join it by referendum. Switzerland maintains diplomatic relations with almost all countries and historically has served as an intermediary between other states. Switzerland is not a member of the European Union; the Swiss people have consistently rejected membership since the early 1990s. However, Switzerland does participate in the Schengen Area."
    answer = "since the end of its expansion in 1515"

    X = "been neutral"
    Y = "Switzerland"
    Z = ""
    Exp = "X and Y appear both in the question and in the context. The answer directly follows X. The answer starts with \"since\"."

    context = context.replace("``", "\"").replace("''", "\"").replace("<font color=\\\"red\\\"> ", "").replace(" </font>", "").replace("<font color=\"red\">", "").replace("</font>", "")
    answer = answer.strip('!\'*+,-./:;<=>?@[\\]^_`{|}~')
    squad_instance = squad_reader.read_one(context, question, answer)

    # create a new ansFunc instance
    ansFunc = AnsFunc()
    ansFunc.instantiate(squad_instance)
    ansFunc.add_variable(Variable.get_variable(squad_instance, answer, 'Answer'))

    Exp = Exp.strip(' ')
    # Force X,Y,Z must appear
    if len(X) > 0 and (X.lower() in question.lower() or X.lower() in context.lower()):
        ansFunc.add_variable(Variable.get_variable(squad_instance, X, 'X'))
        Exp += ' X appears in the context.'
    if len(Y) > 0 and (Y.lower() in question.lower() or Y.lower() in context.lower()):
        ansFunc.add_variable(Variable.get_variable(squad_instance, Y, 'Y'))
        Exp += ' Y appears in the context.'
    if len(Z) > 0 and (Z.lower() in question.lower() or Z.lower() in context.lower()):
        ansFunc.add_variable(Variable.get_variable(squad_instance, Z, 'Z'))
        Exp += ' Z appears in the context.'

    # prepare input
    inputs = {'Answer': {'span': answer,
                         'begin_idx': squad_instance.start_position[0],
                         'end_idx': squad_instance.end_position[0],
                         'sentence_idx': squad_instance.idx_sentence_containing_answer[0]},
              'X': X,
              'Y': Y,
              'Z': Z,
              'all': question + ' ' + context,
              'instance': squad_instance,
              'pretrained_modules': None,
              'soft': False,
              }

    # preprocess explanation
    sentences = preprocess_sent(Exp)
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    print(tokenized_sentences)

    # validate each rule on the reference instance
    for sentence in tokenized_sentences:
        rule_list_sentence = []
        for potential_sentence in sentence:
            all_possible_parses = parser.parse(potential_sentence)
            if len(all_possible_parses) > 0:
                rule_list_sentence += all_possible_parses
        if len(rule_list_sentence) > 0:
            for a_rule in rule_list_sentence:
                print(a_rule)
                rule = Rule(a_rule)
                inputs_copy = copy.copy(inputs)
                if np.equal(1, rule.execute(inputs_copy)):
                    ansFunc.add_rule(a_rule)
                    break

    print([item.raw_str for item in ansFunc.all_rules])
    all_answers = ansFunc.answer(squad_instance, pretrained_modules=None, soft=False, thres=1.0)

    answer_tokenized = ' '.join([item.text for item in squad_instance.answer_tokens[0]])
    if len(all_answers) == 1:
        one_answer = all_answers[0]
        validated = exact_match(one_answer['Answer'].span, answer_tokenized)

    print('Validated?: {}. Produced Answer: {}'.format(validated, [one_answer['Answer'].span for one_answer in all_answers]))

if __name__ == '__main__':
    main()