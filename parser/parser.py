from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

import copy
import logging

from dictionary import STRING2PREDICATE, WORD2NUMBER, RAW_LEXICON
from constant import MAX_PHRASE_LEN, BEAM_WIDTH, SPECIAL_CHARS, REVERSE_SPECIAL_CHARS
from nltk.ccg import chart, lexicon

nlp = English()
tokenizer = Tokenizer(nlp.vocab)

logger = logging.getLogger(__name__)

### Helper functions for Preprocessing Explanations ###

def fill_whitespace_in_quote(sentence):
    """input: a string containing multiple sentences;
    output: fill all whitespaces in a quotation mark into underscore"""

    def convert_special_chars(s, flag):
        return SPECIAL_CHARS[s] if s in SPECIAL_CHARS and flag else s

    flag = False  # whether space should be turned into underscore, currently
    output_sentence = ''
    for i in range(len(sentence)):
        if sentence[i] == "\"":
            flag = not flag  # flip the flag if a quote mark appears
        output_sentence += convert_special_chars(sentence[i], flag)
    return output_sentence


def preprocess_sent(sentence):
    """input: a string containing multiple sentences;
    output: a list of tokenized sentences"""
    sentence = fill_whitespace_in_quote(sentence)
    output = tokenizer(sentence)
    tokens = list(map(lambda x: x.text, output))
    ret_sentences = []
    st = 0

    # fix for ','
    new_tokens = []
    for i, token in enumerate(tokens):
        if token.endswith(','):
            new_tokens += [token.rstrip(','), ',']
        else:
            new_tokens += [token]
    tokens = new_tokens

    for i, token in enumerate(tokens):
        if token.endswith('.'):
            ret_sentences.append(tokens[st: i] + [token.strip('.')])
            st = i + 1
    return ret_sentences


def string_to_predicate(s):
    """input: one string (can contain multiple tokens with ;
    output: a list of predicates."""
    if s != ',' and s not in REVERSE_SPECIAL_CHARS:
        s = s.lower().strip(',')
    if s.startswith("$"):
        return [s]
    elif s.startswith("\"") and s.endswith("\""):
        return ["'" + s[1:-1] + "'"]
    elif s in STRING2PREDICATE:
        return STRING2PREDICATE[s]
    elif s.isdigit():
        return ["'" + s + "'"]
    elif s in WORD2NUMBER:
        return ["'" + WORD2NUMBER[s] + "'"]
    else:
        return []


def tokenize(sentence):
    """input: a list of tokens;
    output: a list of possible tokenization of the sentence;
    each token can be mapped to multiple predicates"""
    # log[j] is a list containing temporary results using 0..(j-1) tokens
    log = {i: [] for i in range(len(sentence) + 1)}
    log[0] = [[]]
    for i, token in enumerate(sentence):
        for _range in range(1, MAX_PHRASE_LEN + 1):
            if i + _range > len(sentence):
                break
            phrase = ' '.join(sentence[i:i + _range])
            predicates = string_to_predicate(phrase)
            for temp_result in log[i]:
                for predicate in predicates:
                    log[i + _range].append(temp_result + [predicate])
            if token.startswith("\""):  # avoid --"A" and "B"-- treated as one predicate
                break
    return log[len(sentence)]


def get_word_name(layer, st, idx):
    return "$Layer{}_St{}_{}".format(str(layer), str(st), str(idx))

def get_entry(word_name, category, semantics):
    return "\n\t\t{0} => {1} {{{2}}}".format(word_name, str(category), str(semantics))


### Helper functions for Parsing ###

def quote_word_lexicon(sentence):
    """Special Handle for quoted words"""

    def is_quote_word(token):
        return (token.startswith("\'") and token.endswith("\'")) \
            or (token.startswith("\"") and token.endswith("\""))

    ret = ""
    for token in sentence:
        if is_quote_word(token):
            ret += get_entry(token, 'NP', token)
            ret += get_entry(token, 'N', token)
            ret += get_entry(token, 'NP', "'@In'({},'all')".format(token))
            if token[1:-1].isdigit():
                ret += get_entry(token, 'NP/NP', "\\x.'@Num'({},x)".format(token))
                ret += get_entry(token, 'N/N', "\\x.'@Num'({},x)".format(token))
                ret += get_entry(token, 'PP/PP/NP/NP', "\\x y F.'@WordCount'('@Num'({},x),y,F)".format(token))
                ret += get_entry(token, 'PP/PP/N/N', "\\x y F.'@WordCount'('@Num'({},x),y,F)".format(token))

    return ret


### Main Class for Parser ###

class Parser():
    def __init__(self):
        super(Parser, self).__init__()
        self.raw_lexicon = RAW_LEXICON
        self.beam_width = BEAM_WIDTH

    def parse(self, sentence, beam=True):
        """
        :param sentence: a list of tokens in one sentence.
                e.g. ['"may_be"', '$Is', '$Between', '$ArgX', '$And', '$ArgY']
        :return: a list of successful parses.
        """
        beam_lexicon = copy.deepcopy(self.raw_lexicon) + quote_word_lexicon(sentence)

        # the first index of forms is layer
        # the second index of forms is starting index
        all_forms = [[[token] for token in sentence]]

        # parsed results to be returned
        ret = []

        # Width of tokens to be parsed. Start with width 1 and stack to len(sentence)
        for layer in range(1, len(sentence)):
            layer_form = []

            # update the lexicon from previous layers
            lex = lexicon.fromstring(beam_lexicon, True)
            parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)

            # parse the span (st, st+layer)
            for st in range(0, len(sentence) - layer):
                form = []
                memory = []  # keep a memory and remove redundant parses
                word_index = 0
                ed = st + layer
                # try to combine (st, split), (split+1, ed) into (st, ed)
                for split in range(st, ed):

                    # get candidates for (st, split) and (split+1, ed)
                    words_L = all_forms[split-st][st]
                    words_R = all_forms[ed-split-1][split+1]

                    for word_L in words_L:
                        for word_R in words_R:
                            # try to combine word_L and word_R
                            try:
                                for parse in parser.parse([word_L, word_R]):
                                    token, _ = parse.label()
                                    category, semantics = token.categ(), token.semantics()
                                    memory_key = str(category) + '_' + str(semantics)
                                    if memory_key not in memory:
                                        memory.append(memory_key)
                                        word_index += 1
                                        form.append((parse, category, semantics, word_index))
                            except (AssertionError, SyntaxError) as e:
                                logger.info('Error when parsing {} and {}'.format(word_L, word_R))
                                logger.info('Error information: {}'.format(e.args))

                to_add = []
                for item in form:
                    parse, category, semantics, word_index = item
                    word_name = get_word_name(layer, st, word_index)
                    to_add.append(word_name)
                    beam_lexicon += get_entry(word_name, category, semantics)

                    # if this is the last layer (covering the whole sentence)
                    # add this to output
                    if layer == len(sentence) - 1:
                        ret.append(str(semantics))
                layer_form.append(to_add)

            all_forms.append(layer_form)

        # filter incomplete parses
        ret = list(filter(lambda x: x.startswith("'@"), ret))
        return list(ret)

def unit_test():
    sent = '"may be" is between X and Y. The answer is right after Y. The answer starts with "by".'

    # Split the long explanation into a list of sentences.
    sentences = preprocess_sent(sent)

    # For each sentence try to tokenize according to our lexicon dict.
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    parser = Parser()

    print('=' * 20 + ' start parsing ' + '=' * 20 + '\n')

    for i, sentence in enumerate(tokenized_sentences):
        print('=== sentence {}: {}'.format(i, sentences[i]))
        rule_list_sentence = []
        for potential_sentence in sentence:
            print('sentence predicates: {}'.format(potential_sentence))
            all_possible_parses = parser.parse(potential_sentence)
            if len(all_possible_parses) > 0:
                rule_list_sentence += all_possible_parses
                print('parses: {}\n'.format(all_possible_parses))

if __name__ == "__main__":
    unit_test()