import json
import time
import logging
import pickle
from tqdm import tqdm

from utils.squad_utils import SQuADExampleExtended

from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import char_span_to_token_span

# from support.utils_squad import SQuADExampleExtended

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
HEADER = ['< Table >', '< Tr >', '< Ul >', '< Ol >', '< Dl >', '< Li >', '< Dd >', '< Dt >', '< Th >', '< Td >']
class SquadReader:
    def __init__(self, pre=True) -> None:
        self._tokenizer = WordTokenizer()
        self._sentence_splitter = SpacySentenceSplitter()
        if pre:
            self._constituency_parser = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
            self._ner_tagger = Predictor.from_path("https://allennlp.s3.amazonaws.com/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
            self._dependency_parser = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
            # self._constituency_parser._model = self._constituency_parser._model.cuda()
            # self._ner_tagger._model = self._ner_tagger._model.cuda()
            # self._dependency_parser._model = self._dependency_parser._model.cuda()

    def preprocess(self, file_path: str):
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")

        for article in tqdm(dataset):
            for idx, paragraph_json in enumerate(tqdm(article['paragraphs'])):
                paragraph = paragraph_json["context"]

                sentences = self._sentence_splitter.split_sentences(paragraph)
                passage_info = []
                offset = 0 # token offset
                for sentence in sentences:
                    tokens = self._tokenizer.tokenize(sentence)
                    constituency = self._constituency_parser.predict(sentence=sentence)
                    ners = self._ner_tagger.predict(sentence=sentence)
                    dependency = self._dependency_parser.predict(sentence=sentence)
                    passage_info.append(self.get_info(tokens, constituency, ners, dependency, offset))
                    offset += len(tokens)

                passage_tokens = self._tokenizer.tokenize(paragraph)
                passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
                paragraph_json["paragraph_offsets"] = passage_offsets
                paragraph_json["context_info"] = passage_info

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    question_answer["question_info"] = self.get_info(self._tokenizer.tokenize(question_text),
                                           self._constituency_parser.predict(sentence=question_text),
                                           self._ner_tagger.predict(sentence=question_text),
                                           self._dependency_parser.predict(sentence=question_text), 0)

                    for answer in question_answer['answers']:
                        text = answer['text']
                        span_starts = answer['answer_start']
                        span_ends = span_starts + len(text)
                        if span_ends > passage_offsets[-1][1]:
                            answer['token_start'], answer['token_end'] = -1, -1
                            print(text)
                            print(paragraph)
                            print('-' * 20)
                        else:
                            (span_start, span_end), error = char_span_to_token_span(passage_offsets, (span_starts, span_ends))
                            answer['token_start'], answer['token_end'] = span_start, span_end

        return dataset

    def read_proprocessed(self, file_path: str):
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            # dataset = dataset_json['data']
        logger.info("Reading the dataset")

        all_instances = []

        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                paragraph = paragraph_json["context"]
                context_info = paragraph_json["context_info"]
                for idx, sentence in enumerate(context_info):
                    sentence['location'], sentence['sentence_idx'] = 'Context', idx

                doc_tokens = []
                for sentence in context_info:
                    doc_tokens += sentence['tokens']
                if len(doc_tokens) + len(context_info) > 510: # too long for bert encoder!
                    # logger.info("Discard a paragraph because it is too long. {}".format(paragraph_json["context"]))
                    continue
                doc_tokens = context_info

                paragraph_offsets = paragraph_json["paragraph_offsets"]

                for question_answer in paragraph_json["qas"]:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    question_info = question_answer["question_info"]
                    question_info['location'], question_info['sentence_idx'] = 'Question', 0
                    is_impossible = len(question_answer['answers']) > 0

                    try:
                        answer_texts = [answer['text'] for answer in question_answer['answers']]
                        answer_tokens = [self._tokenizer.tokenize(answer['text']) for answer in question_answer['answers']]
                        span_starts = [answer['token_start'] for answer in question_answer['answers']]
                        span_ends = [answer['token_end'] for answer in question_answer['answers']]
                        idx_sentence_containing_answer = [self.get_idx_sentence_containing_answer(context_info, answer['token_start'])
                                                          for answer in question_answer['answers']]
                        qas_id = question_answer.get("id", "")


                        instance = SQuADExampleExtended(qas_id=qas_id,
                                                        question_text=question_text,
                                                        doc_tokens=doc_tokens,
                                                        orig_answer_text=answer_texts,
                                                        start_position=span_starts,
                                                        end_position=span_ends,
                                                        is_impossible=is_impossible,
                                                        paragraph_offsets=paragraph_offsets,
                                                        context_info=context_info,
                                                        question_info=question_info,
                                                        original_paragraph=paragraph,
                                                        answer_tokens=answer_tokens,
                                                        idx_sentence_containing_answer=idx_sentence_containing_answer,
                                                        weight=1.0
                                                        )

                        if instance is not None:
                            # yield instance
                            all_instances.append(instance)
                    except Exception as e:
                        print(e.args)
                        pass

        return all_instances

    def get_idx_sentence_containing_answer(self, passage_info, span_start):
        idx_sentence_containing_answer = -1
        for i, item in enumerate(passage_info):
            if span_start < item['offset']:
                idx_sentence_containing_answer = i - 1
                break
        if idx_sentence_containing_answer == -1:
            idx_sentence_containing_answer = len(passage_info) - 1
        return idx_sentence_containing_answer


    def read_one(self, paragraph, question, answer):
        """read one instance and transform to our form"""

        sentences = self._sentence_splitter.split_sentences(paragraph)
        passage_info = []
        offset = 0
        for idx, sentence in enumerate(sentences):
            tokens = self._tokenizer.tokenize(sentence)
            constituency = self._constituency_parser.predict(sentence=sentence)
            ners = self._ner_tagger.predict(sentence=sentence)
            dependency = self._dependency_parser.predict(sentence=sentence)
            temp = self.get_info(tokens, constituency, ners, dependency, offset)
            temp['location'] = 'Context'
            temp['sentence_idx'] = idx
            passage_info.append(temp)
            offset += len(tokens)



        passage_tokens = self._tokenizer.tokenize(paragraph)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        question_text = question.strip().replace("\n", "")
        question_info = self.get_info(self._tokenizer.tokenize(question_text),
                                                         self._constituency_parser.predict(sentence=question_text),
                                                         self._ner_tagger.predict(sentence=question_text),
                                                         self._dependency_parser.predict(sentence=question_text), 0)
        question_info['location'] = 'Question'
        question_info['sentence_idx'] = 0

        span_start, span_end, answer_tokens, idx_sentence_containing_answer = 0, 0, None, -1
        if len(answer) > 0:
            answer_tokens = self._tokenizer.tokenize(answer)
            span_starts = paragraph.index(answer)
            span_ends = span_starts + len(answer)
            (span_start, span_end), error = char_span_to_token_span(passage_offsets, (span_starts, span_ends))

            idx_sentence_containing_answer = self.get_idx_sentence_containing_answer(passage_info, span_start)

        return SQuADExampleExtended(qas_id='0',
                                    question_text=question,
                                    doc_tokens=passage_tokens,
                                    orig_answer_text=[answer],
                                    start_position=[span_start],
                                    end_position=[span_end],
                                    is_impossible=False,
                                    paragraph_offsets=passage_offsets,
                                    context_info=passage_info,
                                    question_info=question_info,
                                    original_paragraph=paragraph,
                                    answer_tokens=[answer_tokens],
                                    idx_sentence_containing_answer=[idx_sentence_containing_answer],
                                    weight=1.0
                                    )

    def get_info(self, _tokens, constituency, ner, dependency, offset):
        tokens = [token.text for token in _tokens]
        lemmas = [token.lemma_ for token in _tokens]
        constituency_output = self.traverse_tree(constituency['hierplane_tree']['root'], tokens, 0, len(tokens))
        dependency_output = {'pos': dependency['pos'],
                             'predicted_dependencies': dependency['predicted_dependencies'],
                             'predicted_heads': dependency['predicted_heads']}
        return {'tokens': tokens,
                'lemmas': lemmas,
                'constituency': constituency_output,
                'ner_seq': ner['tags'],
                'ner': self.ner_spans(ner['words'], ner['tags']),
                'dependency': dependency_output,
                'offset': offset}

    def ner_spans(self, tokens, predicted_tags):
        predicted_spans = []
        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            # if its a U, add it to the list
            if tag[0] == 'U':
                current_tags = {'span': ' '.join(tokens[i: i + 1]),
                                'begin_idx': i,
                                'end_idx': i,
                                'tag': tag.split('-')[1]
                                }
                predicted_spans.append(current_tags)
            # if its a B, keep going until you hit an L.
            elif tag[0] == 'B':
                begin_idx = i
                while tag[0] != 'L':
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i
                current_tags = {'span': ' '.join(tokens[begin_idx: end_idx + 1]),
                                'begin_idx': begin_idx,
                                'end_idx': end_idx,
                                'tag': tag.split('-')[1]
                                }
                predicted_spans.append(current_tags)
            i += 1

        return predicted_spans

    def find_idx(self, word, tokens, st, ed):
        for i in range(st, ed):
            for j in range(i, ed):
                if ' '.join(tokens[i:j+1]) == word:
                    return i, j+1
        return -1, -1

    def traverse_tree(self, tree, tokens, st, ed):
        st1, ed1 = self.find_idx(tree['word'], tokens, st, ed)
        ret = [{'span': tree['word'],
                'tag': tree['nodeType'],
                'begin_idx': st1,
                'end_idx': ed1 - 1}]
        if 'children' in tree:
            for child in tree['children']:
                ret += self.traverse_tree(child, tokens, st1, ed1)
        return ret


def get_sent_idx(st, ed, offsets):
    for i in range(1, len(offsets)):
        if offsets[i - 1] <= st <= ed < offsets[i]:
            return i  # question -> -1; context_sentence 0 -> 0
    return -1  # not found, error

def preprocess_all():
    start_time = time.time()
    reader = SquadReader()
    instances = reader.preprocess('./albert/dataset/squad/test_new.json')

    out_file = './albert/dataset/squad/test_new_processed_dep.json'
    with open(out_file, 'w') as fout:
        json.dump(instances, fout)

    elapsed_time = time.time() - start_time
    print(elapsed_time)

def read_preprocessed():
    start_time = time.time()
    reader = SquadReader(pre=False)
    instances = reader.read_proprocessed('./data/squad/train_processed_dep.json')

    # print(instances[0].context_info)
    print(instances[0].question_info)
    print(instances[0].doc_tokens)
    print(instances[0].orig_answer_text)
    print(len(instances))
    elapsed_time = time.time() - start_time
    print(elapsed_time)

if __name__ == "__main__":
    read_preprocessed()



