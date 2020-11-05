import re
import copy
import logging
import torch
import numpy as np
from itertools import product

from allennlp.data.tokenizers import WordTokenizer

from module.modules import OPS, fill, _and
from utils.squad_utils import SQuADExampleExtended, normalize_answer
from utils.squad_reader import SquadReader

from constant import CHUNK_DICT, NER_DICT, VAR_NAMES
from dictionary import QUESTION_WORDS

tokenizer = WordTokenizer()

logger = logging.getLogger(__name__)

class Rule:
    """Same as LF in NMET."""

    def __init__(self, str_rep=""):
        # This part is using NMN
        # String Representation
        self.raw_str = str_rep
        self.tree = ('.root',)
        # self.parsed = self.parse(str_rep)
        # Executable Function
        try:
            self.tree = self.str2tree(str_rep)
            self.func = self.recurse(self.tree)
        except:
            logging.warning("Rule not executable: {}".format(self.tree))
            self.func = Rule.return_false

    @classmethod
    def return_false(cls, *argv):
        return False

    @classmethod
    def return_true(cls, *argv):
        return True

    def __str__(self):
        return self.raw_str

    @classmethod
    def str2tree(cls, str_rep):
        sem = list(filter(lambda i: i, re.split(',|(\()', str_rep)))
        for idx, w in enumerate(sem):
            if w == '(' and idx > 0:
                sem[idx], sem[idx-1] = sem[idx-1], sem[idx]
        sem_ = ''
        for i, s in enumerate(sem):
            if sem[i-1] != '(':
                sem_ += ','
            sem_ += sem[i]
        try:
            em = ('.root', eval(sem_[1:]))
        except SyntaxError as e:
            logging.info('Error when transforming'.format(str_rep))
            logging.info('Error information: {}'.format(e.args))
            em = ('.root',)
        return em

    @classmethod
    def recurse(cls, sem):
        # print(sem)
        if isinstance(sem, tuple):
            if sem[0] not in OPS:
                logging.info('Error: {} not implemented'.format(sem[0]))
                raise NotImplementedError
            op = OPS[sem[0]]
            args = [cls.recurse(arg) for arg in sem[1:]]
            return op(*args) if args else op
        else:
            return sem

    def execute(self, args):
        # give args and output score
        try:
            ret = self.func(args)
        except (TypeError, AttributeError, ValueError, AssertionError) as e:
            logging.info('Error in executing the rule: {}'.format(self.tree))
            logging.info('Error information: {}'.format(e.args))
            ret = 0.0
        return ret

    def clean_rules(self):
        self.func = None

    def reload_rules(self):
        try:
            self.func = self.recurse(self.tree)
        except:
            logging.info("Rule not executable: {}".format(self.tree))
            self.func = Rule.return_false


class Variable:
    """One variable in the rule.
    e.g. X(NP)=packet switching; Y(VBZ)=characterize; Z(NP,LOCATION)=USC
    """

    def __init__(self, name, chunk=None, ner=None, value=None):
        self.name = name
        self.chunk = chunk
        self.ner = ner
        self.value = value
        self.dependency = []
        self.candidates = []
        self.in_question = False
        self.loc_in_question = []
        self.in_context = False
        self.loc_in_context = []

    @classmethod
    def get_variable(cls, instance: SQuADExampleExtended, value, name):

        def remove_repeated(lst):
            return list(filter(lambda x: x, set(lst)))

        def get_tag(lst, key):
            ret = []
            for item in lst:
                t = ' '.join(item['span']) if isinstance(item['span'], list) else item['span']
                if normalize_answer(t) == normalize_answer(key) and len(normalize_answer(t)) > 0:
                    ret.append(item)
            return ret

        def locate(value, tokens, lemmas, offset):
            ret = []
            for st in range(len(tokens)):
                for ed in range(st+1, len(tokens) + 1):
                    if ' '.join(value).lower() == ' '.join(lemmas[st:ed]).lower():
                        ret.append({'st': st + offset, 'ed': ed + offset,
                                    'st_in_sentence': st, 'ed_in_sentence': ed,
                                    'original': ' '.join(tokens[st:ed])})
            return ret

        def dependency_info(dep_dict, st, ed):
            old_heads = dep_dict['predicted_heads'][st:ed]
            new_heads = [item-1-st if st<=item-1<ed else -1 for item in old_heads]
            return {
                'pos': dep_dict['pos'][st:ed],
                'dependencies' : dep_dict['predicted_dependencies'][st:ed],
                'heads': new_heads,
                'offset': st,
                'len': ed - st
            }

        variable = Variable(name)

        chunk_tag = []
        ner_tag = []
        dependency = []
        value_tokenized = tokenizer.tokenize(value)
        value_lemmas = [token.lemma_ for token in value_tokenized]

        locations_in_question = locate(value_lemmas,
                                       instance.question_info['tokens'],
                                       instance.question_info['lemmas'],
                                       offset=0)
        if len(locations_in_question) > 0:
            variable.in_question = True
            for item in locations_in_question:
                chunk_tag += get_tag(instance.question_info['constituency'], item['original'])
                ner_tag += get_tag(instance.question_info['ner'], item['original'])
                if 'dependency' in instance.question_info:
                    dependency.append(dependency_info(instance.question_info['dependency'], item['st_in_sentence'], item['ed_in_sentence']))

        idx = instance.idx_sentence_containing_answer[0]
        sentence = instance.context_info[idx]
        locations_in_sentence = locate(value_lemmas,
                                       sentence['tokens'],
                                       sentence['lemmas'],
                                       offset=sentence['offset'])
        if len(locations_in_sentence) > 0:
            variable.in_context = True
            for item in locations_in_sentence:
                chunk_tag += get_tag(sentence['constituency'], item['original'])
                ner_tag += get_tag(sentence['ner'], item['original'])
                if 'dependency' in sentence:
                    dependency.append(dependency_info(sentence['dependency'], item['st_in_sentence'], item['ed_in_sentence']))

        variable.chunk = chunk_tag
        variable.ner = ner_tag
        variable.value = value
        variable.lemmas = value_lemmas
        variable.loc_in_context = locations_in_sentence
        variable.loc_in_question = locations_in_question
        variable.dependency = dependency
        return variable


class AnsFunc:
    def __init__(self):
        # Variables used in the Func
        # e.g. X(NP), Y(VBZ), ANS(PP)
        self.variables = {'Answer': Variable(name='Answer')}

        self.all_rules = []

        # The QA instance that this AnsFunc is extracted from
        self.reference_instance = None

        # Question Head
        self.question_word = ""
        self.question_head = ""

        # Variable order in question
        # self.question_order = []

        # Variable order in context
        # self.context_order = []

    def all_rules_str(self):
        return ','.join([item.raw_str for item in self.all_rules])

    def instantiate(self, instance):

        def find_match(query, keys):
            for key in keys:
                if re.search('^' + key + '$', query):
                    return key
            return None

        def get_question_attribute(question):
            lemmas = [token.lower() for token in question['lemmas']]
            lemmas_str = ' '.join(lemmas)
            for idx, word in enumerate(lemmas):
                key = find_match(word, QUESTION_WORDS)
                if key:
                    if 'of what' in lemmas_str:
                        return word, 'of what'
                    elif 'in which' in lemmas_str:
                        return word, 'in which'
                    # elif 'what' in lemmas_str and 'what be' not in lemmas_str:
                    #     return word, word
                    elif 'who' in lemmas_str:
                        return word, word
                    else:
                        return word, ' '.join(lemmas[idx: idx+2])
            return '', ''

        self.reference_instance = instance
        self.question_word, self.question_head = get_question_attribute(instance.question_info)

    def clean_vars(self):
        for var in self.variables.values():
            var.value = ""
            var.candidates = []
            var.loc_in_question = -1
            var.loc_in_context = -1

    def answer(self, instance: SQuADExampleExtended, pretrained_modules=None, soft=False, thres=1.0):
        """
        :param instance: a SQuADExampleExtended instance
        :return: answer, confidence
        """
        def get_key(state):
            key = ''
            for var_name in VAR_NAMES:
                item = state[var_name] if var_name in state else None
                if item:
                    key += ','.join([var_name, str(item.begin_idx), str(item.end_idx),
                                     str(item.location), str(item.sentence_idx)]) + '_'
            key += ',context_idx_' + str(state['context_idx'])
            return key

        def filter_states(list_of_states, soft):
            unique_keys = []
            ret_list = []
            for state in list_of_states:
                key = get_key(state)
                if key not in unique_keys:
                    unique_keys.append(key)
                    ret_list.append(state)
            if soft:
                ret_list = sorted(ret_list, key=lambda x: x['confidence'], reverse=True)
                ret_list = ret_list[:8]
            return ret_list

        def filter_answers(list_of_states):
            unique_keys = []
            ret_list = []
            for item in list_of_states:
                answer = item['Answer']
                answer_norm = normalize_answer(answer.span)
                question_text = item['instance'].question_text
                key = ','.join([answer_norm,
                                     str(answer.location), str(answer.sentence_idx)])
                if key not in unique_keys and \
                                answer_norm not in question_text and \
                                answer.span not in question_text:
                    unique_keys.append(key)
                    ret_list.append(item)
            return ret_list

        # Question head
        if len(self.question_head) > 0:
            question_head = self.question_head
            question_lemmatized = ' '.join(instance.question_info['lemmas']).lower()
            if not question_head in question_lemmatized:
                return []

        # Fill candidates for each variable
        # E.g. X.candidates = [x1, x2]; Y.candidates = [y1, y2, y3];
        for var in self.variables.values():
            var.candidates = fill(self.reference_instance.question_info, var, instance.question_info, pretrained_modules, soft)

        ## Initialize states
        initial_state = {k: None for k in self.variables}
        initial_state['instance'] = instance
        initial_state['soft'] = soft
        initial_state['pretrained_modules'] = pretrained_modules
        thres = thres if soft else 1.0

        var_list = [key for key in self.variables.keys() if key != 'Answer'] + ['Answer']

        all_answer = []

        for idx, sentence in enumerate(instance.context_info): 
            # try to find answer in each sentence in the paragraph

            new_state = copy.copy(initial_state)
            new_state['Context'] = sentence
            new_state['context_idx'] = idx
            new_state['all'] = {'tokens': sentence['tokens'] + instance.question_info['tokens'],
                                'lemmas': sentence['lemmas'] + instance.question_info['lemmas']}

            prev_states = [new_state]

            # Fill answer candidate for question.
            var = self.variables['Answer']
            context_idx = self.reference_instance.idx_sentence_containing_answer[0]
            context_sentence = self.reference_instance.context_info[context_idx]
            var.candidates = fill(context_sentence, var, sentence, pretrained_modules, soft)

            for j in var_list: # fill (X, Y, Z, Ans) step by step
                new_states = []
                for state in prev_states:
                    candidates = self.variables[j].candidates
                    for candidate in candidates:
                        # attempt to fill with current candidate
                        a_new_state = copy.copy(state)
                        a_new_state[j] = candidate

                        # try to evaluate the current combination of candidates
                        confidence_for_this_new_state = self.eva_state(a_new_state, soft)
                        if isinstance(confidence_for_this_new_state, torch.Tensor):
                            confidence_for_this_new_state_scalar = confidence_for_this_new_state.item()
                        else:
                            confidence_for_this_new_state_scalar = float(confidence_for_this_new_state)
                        assert np.less_equal(confidence_for_this_new_state_scalar, 1.0)

                        # if confidence is greater than a threshold, keep the current state
                        if np.greater_equal(confidence_for_this_new_state_scalar, thres):
                            a_new_state['confidence'] = confidence_for_this_new_state
                            new_states.append(a_new_state)

                new_states = filter_states(new_states, soft)
                prev_states = new_states

            all_answer += filter_answers(prev_states)

        return all_answer

    def eva_state(self, inputs, soft=False):
        """
        inputs may be partially filled; only some of the rules can be evaluated.
        select these rules and evaluate
        :return: Boolean
        """
        def vars_needed(str_rep, var_names):
            ret = []
            for var_name in var_names:
                if '\'' + var_name + '\'' in str_rep:
                    ret.append(var_name)
            return set(ret)

        def overlap(st1, ed1, st2, ed2):
            return st1 <= st2 <= ed1 or st1 <= ed2 <= ed1 or \
                   st2 <= st1 <= ed2 or st2 <= ed1 <= ed2

        filled_vars = [item[0] for item in filter(lambda x: x[1], inputs.items())]

        # Check the variables are not the same
        for i in filled_vars:
            for j in filled_vars:
                if i != j and i in VAR_NAMES and j in VAR_NAMES:
                    if inputs[i].span in inputs[j].span or inputs[j].span in inputs[i].span:
                        return False

        probs_list = []
        for rule in self.all_rules:
            vars_needed_for_this_rule = vars_needed(rule.raw_str, VAR_NAMES)
            if vars_needed_for_this_rule.issubset(set(filled_vars)):
                probs_list.append(rule.execute(inputs))

        for var in filled_vars:
            if var in VAR_NAMES:
                probs_list.append(inputs[var].confidence)

        # the prob_list contain (1) rule execution confidence scores (2) variable filling confidence.
        # use a top-level _and to aggregate them.

        return _and(probs_list, soft)

    def add_variable(self, v: Variable):
        """
        :param v: a variable to be added to the dict
        """
        name = v.name
        self.variables[name] = v

    def add_rule(self, exp):
        self.all_rules.append(Rule(exp))

    def clean_rules(self):
        """To save to pkl file, the instance cannot include any function."""
        for rule in self.all_rules:
            rule.clean_rules()

    def reload_rules(self):
        """Load pkl file and recover the functions"""
        for rule in self.all_rules:
            rule.reload_rules()

    def delete_redundant_rules(self):
        var_list = list(self.variables.keys())
        var_list.remove('Answer')
        n_all_rules = len(self.all_rules)
        # var_list = [item for item in ['X', 'Y', 'Z'] if item in self.variables.keys()]
        for idx0, var in enumerate(var_list[::-1]):
            kept_idx = n_all_rules - (idx0 + 1)
            temp_rule = self.all_rules[kept_idx]
            for rule in self.all_rules[:-len(var_list)]:
                if var in rule.raw_str and 'Question' not in rule.raw_str:
                    self.all_rules.pop(kept_idx)
                    break

def unit_test():
    
    # "in" is between X and Y
    # "one year" is within 5 words right of X
    str_rep = "'@And'('@Is'('in','@Between'('@And'('X','Y'))),'@Is'('one year','@LessThan'('@Right'('X'),'@Num'('5','tokens'))))"
    # Construct rule from parsed result
    r1 = Rule(str_rep)

    # create input instance
    squad_reader = SquadReader(pre=True)
    instance = squad_reader.read_one(
        paragraph="I have lived in Los Angeles for one year.",
        question="How long have I lived in Los Angeles",
        answer="for one year",
    )
    inputs = {'Answer': {'span': "for one year",
                        'begin_idx': 0,
                        'end_idx': 0,
                        'sentence_idx': 0},
            'X': "lived",
            'Y': "Los Angeles",
            'Z': "",
            'all': "",
            'instance': instance,
            'pretrained_modules': None,
            'soft': False,
            }

    # execute rule
    ret = r1.func(inputs)
    print(ret)

def unit_test_2():
    pass


if __name__ == "__main__":
    unit_test()