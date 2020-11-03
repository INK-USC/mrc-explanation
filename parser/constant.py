BEAM_WIDTH = 100
MAX_PHRASE_LEN = 4

COMMA_INDEX = {',': 0, '-LRB-': 1, '-RRB-': 2, '.': 3, '-': 4}
SPECIAL_CHARS = {' ': '_', '(': '[LEFT_BRACKET]', ')': '[RIGHT_BRACKET]', '.': '[DOT]', ',': '[COMMA]', '-': '[HYPHEN]', '\'': '[APOSTROPHE]'}
REVERSE_SPECIAL_CHARS = {v.lower(): k for k, v in SPECIAL_CHARS.items()}

CHUNK_DICT = {
    'N': ['N', 'NP', 'NN', 'NNS', 'NNP', 'NNPS'],
    'V': ['VP', 'VB', 'VBD', 'VBG', 'VBN',  'VBP', 'VBZ'],
    'P': ['PP'],
    'ADJ': ['JJ', 'JJR', 'JJS', 'ADJP'],
    'ADV': ['RB', 'RBR', 'RBS', 'ADVP'],
    'NUM': ['CD', 'QP'],
}

NER_DICT = {
    'PERSON': ['PERSON'],
    'NORP': ['NORP'],
    'ORGANIZATION': ['ORG'],
    'GPE': ['GPE'],
    'LOCATION': ['GPE', 'FACILITY', 'ORG', 'LOCATION'],
    'DATE': ['DATE'],
    'TIME': ['DATE', 'TIME'],
    'NUMBER': ['PERCENT', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'MONEY'],
    'PERCENT': ['PERCENT'],
    'MONEY': ['MONEY'],
    'ORDINAL': ['ORDINAL']
}

VAR_NAMES = ['X', 'Y', 'Z', 'Answer']
