import argparse
import pickle
import time
import sys
import numpy as np

import torch.multiprocessing as mp
from multiprocessing import Process, Manager

from allennlp.training.metrics.squad_em_and_f1 import SquadEmAndF1

from utils.general_utils import get_logger
from utils.squad_reader import SquadReader

from rule import AnsFunc, Variable, Rule
from match_utils import get_majority_hard, save_matched

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--ans_func_file", default='./explanations/squad.pkl',
                    type=str, required=False)
arg_parser.add_argument("--qa_file", default='./data/squad/train_processed_dep.json',
                    type=str, required=False)

arg_parser.add_argument("--ratio", default=1.0, type=float, required=False,
                        help="ratio of explanations used in matching (0.3 means first 30% explanations in the file)")
arg_parser.add_argument("--thres", default=1.0, type=float, required=False,
                        help="thres in matching. for strict match this should be 1.0.")
arg_parser.add_argument("--nproc", default=20, type=int, required=False, help='number of process used in matching.')

arg_parser.add_argument('--verbose', action='store_true', default=False, help='If true, update matching progress to stdout.')
arg_parser.add_argument('--save_matched', action='store_true', default=False, help='If true, save the matched instances to out_file')

arg_parser.add_argument("--out_file_hard", default='./data/squad_matched/hard_matched.json',
                    type=str, required=False)
arg_parser.add_argument("--out_file_abstain", default='./data/squad_matched/abstain.json',
                    type=str, required=False)

args = arg_parser.parse_args()
opt = vars(args)

logger = get_logger(__name__, stream_handler=True)

def match_one_instance(instance, ansFuncs, pretrained_modules, opt):

    # in the beginning assume there is no match
    consensus_answer, (st, ed, confidence) = '', (-1, -1, 0.0)

    ans_for_this_instance = []
    rules_used = []
    for idx, ans_func in ansFuncs:
        answers = ans_func.answer(instance, pretrained_modules, soft=False, thres=opt['thres'])

        if len(answers) == 1:
            ans_for_this_instance += answers
            rules_used.append(idx)

    if len(ans_for_this_instance) > 0:
        consensus_answer, (st, ed, confidence) = get_majority_hard(ans_for_this_instance)
    return (consensus_answer, (st, ed, confidence))

def match_multiple_instances(idx0, instances, ansFuncs, return_dict, opt):

    start_time = time.time()

    for idx, ans_func in ansFuncs:
        ans_func.reload_rules()

    pretrained_modules = None

    to_return = []
    for idx, instance in enumerate(instances):
        if idx % 10 == 0 and opt['verbose']:
            sys.stdout.write("Thread {}, processed {} instances , Time: {} sec\r".format(idx0, idx, time.time() - start_time))
            sys.stdout.flush()
        to_return.append(match_one_instance(instance, ansFuncs, pretrained_modules, opt))

    return_dict[idx0] = to_return

def per_instance_match(instances, ansFuncs, opt):

    ### Split instances into n groups
    nproc = opt['nproc']
    instances_per_proc = int(len(instances) / nproc)
    instances_split = [instances[i * instances_per_proc: (i+1) * instances_per_proc] for i in range(nproc - 1)]
    instances_split.append(instances[(nproc-1) * instances_per_proc:])

    ### Initialize Context
    manager = Manager()
    return_dict = manager.dict()

    if nproc > 1:
        procs = []
        for i in range(nproc):
            p = mp.Process(target=match_multiple_instances, args=(i+1, instances_split[i], ansFuncs, return_dict, opt))
            p.start()
            procs.append(p)

        for proc in procs:
            proc.join()
    else:
        return_dict = {}
        match_multiple_instances(1, instances, ansFuncs, return_dict, opt)

    strict_match_instances = []
    abstain_instances = []

    metrics_strict = SquadEmAndF1()
    metrics_all = SquadEmAndF1()

    for idx, predicted in return_dict.items():
        assert len(instances_split[idx-1]) == len(predicted)

        for instance, one_predict in zip(instances_split[idx-1], predicted):
            consensus_answer, (st, ed, confidence) = one_predict
            metrics_all(consensus_answer, instance.orig_answer_text)

            # If the answer is not empty string (strict match)
            if consensus_answer:
                strict_match_instances.append((instance, consensus_answer, (st, ed, confidence)))
                metrics_strict(consensus_answer, instance.orig_answer_text)
            else: # answer is empty string (no match)
                abstain_instances.append((instance, consensus_answer, (st, ed, confidence)))

    logger.info('#Answered: {}'.format(len(strict_match_instances)))
    logger.info('Performance on hard-matched: {}, Count: {}'.format(metrics_strict.get_metric(), metrics_strict._count))
    logger.info('Performance on all-dataset: {}, Count: {}'.format(metrics_all.get_metric(), metrics_all._count))

    if opt['save_matched']:
        save_matched(strict_match_instances, opt['out_file_hard'])
        logger.info("Dumped {} instances to {}".format(len(strict_match_instances), opt['out_file_hard']))
        save_matched(abstain_instances, opt['out_file_abstain'])
        logger.info("Dumped {} instances to {}".format(len(abstain_instances), opt['out_file_abstain']))

def main():

    ans_funcs = pickle.load(open(opt['ans_func_file'], 'rb'))

    if opt['ratio'] < 1.0:
        num_used = round(opt['ratio'] * len(ans_funcs))
        ans_funcs = ans_funcs[:num_used]
    logger.info('Ratio: {}, {} explanations are used.'.format(opt['ratio'], len(ans_funcs)))

    squad_reader = SquadReader(pre=False)
    instances = squad_reader.read_proprocessed(opt['qa_file'])
    logger.info('{} instances loaded from {}.'.format(len(instances), opt['qa_file']))

    per_instance_match(instances=instances, ansFuncs=ans_funcs, opt=opt)

if __name__ == "__main__":
    main()