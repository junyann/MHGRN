import os
import argparse
from multiprocessing import cpu_count
from utils.convert_codah import convert_to_codah_statement
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from utils.grounding import ground
from utils.paths import find_paths, score_paths, prune_paths, generate_path_and_graph_from_adj
from utils.graph import generate_graph, generate_adj_data_from_grounded_concepts
from utils.triples import generate_triples_from_adj

input_paths = {
    'codah': {
        'train': './data/codah/fold_{}/train.jsonl',
        'dev': './data/codah/fold_{}/dev.jsonl',
        'test': './data/codah/fold_{}/test.jsonl',
    },
    'transe': {
        'ent': './data/transe/glove.transe.sgd.ent.npy',
        'rel': './data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'glove': {
        'npy': './data/glove/glove.6B.300d.npy',
        'vocab': './data/glove/glove.vocab',
    },
    'numberbatch': {
        'npy': './data/transe/nb.npy',
        'vocab': './data/transe/nb.vocab',
        'concept_npy': './data/transe/concept.nb.npy'
    },
    'codah': {
        'statement': {
            'train': './data/codah/fold_{}/statement/train.statement.jsonl',
            'dev': './data/codah/fold_{}/statement/dev.statement.jsonl',
            'test': './data/codah/fold_{}/statement/test.statement.jsonl',
            'train-fairseq': './data/codah/fold_{}/fairseq/official/train.jsonl',
            'dev-fairseq': './data/codah/fold_{}/fairseq/official/valid.jsonl',
            'test-fairseq': './data/codah/fold_{}/fairseq/official/test.jsonl',
            'vocab': './data/codah/fold_{}/statement/vocab.json',
        },
        'tokenized': {
            'train': './data/codah/fold_{}/tokenized/train.tokenized.txt',
            'dev': './data/codah/fold_{}/tokenized/dev.tokenized.txt',
            'test': './data/codah/fold_{}/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/codah/fold_{}/grounded/train.grounded.jsonl',
            'dev': './data/codah/fold_{}/grounded/dev.grounded.jsonl',
            'test': './data/codah/fold_{}/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': './data/codah/fold_{}/paths/train.paths.raw.jsonl',
            'raw-dev': './data/codah/fold_{}/paths/dev.paths.raw.jsonl',
            'raw-test': './data/codah/fold_{}/paths/test.paths.raw.jsonl',
            'scores-train': './data/codah/fold_{}/paths/train.paths.scores.jsonl',
            'scores-dev': './data/codah/fold_{}/paths/dev.paths.scores.jsonl',
            'scores-test': './data/codah/fold_{}/paths/test.paths.scores.jsonl',
            'pruned-train': './data/codah/fold_{}/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/codah/fold_{}/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/codah/fold_{}/paths/test.paths.pruned.jsonl',
            'adj-train': './data/codah/fold_{}/paths/train.paths.adj.jsonl',
            'adj-dev': './data/codah/fold_{}/paths/dev.paths.adj.jsonl',
            'adj-test': './data/codah/fold_{}/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/codah/fold_{}/graph/train.graph.jsonl',
            'dev': './data/codah/fold_{}/graph/dev.graph.jsonl',
            'test': './data/codah/fold_{}/graph/test.graph.jsonl',
            'adj-train': './data/codah/fold_{}/graph/train.graph.adj.pk',
            'adj-dev': './data/codah/fold_{}/graph/dev.graph.adj.pk',
            'adj-test': './data/codah/fold_{}/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/codah/fold_{}/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/codah/fold_{}/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/codah/fold_{}/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/codah/fold_{}/triples/train.triples.pk',
            'dev': './data/codah/fold_{}/triples/dev.triples.pk',
            'test': './data/codah/fold_{}/triples/test.triples.pk',
        },
    },
}


def fill_in(dic, fold):
    for key, value in dic.items():
        if isinstance(value, str):
            dic[key] = value.format(fold)
        elif isinstance(value, dict):
            dic[key] = fill_in(value, fold)
        else:
            raise ValueError()
    return dic


def check_paths(dic):
    for key, value in dic.items():
        if isinstance(value, str):
            folder, file_name = os.path.split(value)
            os.makedirs(folder, exist_ok=True)
        elif isinstance(value, dict):
            check_paths(value)
        else:
            raise ValueError()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['codah'], choices=['codah'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=int(0.8 * cpu_count()), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--fold', type=int, default=0, help='enable debug mode')

    args = parser.parse_args()

    global input_paths, output_paths
    input_paths = fill_in(input_paths, args.fold)
    output_paths = fill_in(output_paths, args.fold)
    check_paths(output_paths)

    routines = {
        'codah': [
            {'func': convert_to_codah_statement, 'args': (input_paths['codah']['train'], output_paths['codah']['statement']['train'], output_paths['codah']['statement']['train-fairseq'])},
            {'func': convert_to_codah_statement, 'args': (input_paths['codah']['dev'], output_paths['codah']['statement']['dev'], output_paths['codah']['statement']['dev-fairseq'])},
            {'func': convert_to_codah_statement, 'args': (input_paths['codah']['test'], output_paths['codah']['statement']['test'], output_paths['codah']['statement']['test-fairseq'])},
            {'func': tokenize_statement_file, 'args': (output_paths['codah']['statement']['train'], output_paths['codah']['tokenized']['train'])},
            {'func': tokenize_statement_file, 'args': (output_paths['codah']['statement']['dev'], output_paths['codah']['tokenized']['dev'])},
            {'func': tokenize_statement_file, 'args': (output_paths['codah']['statement']['test'], output_paths['codah']['tokenized']['test'])},
            {'func': make_word_vocab, 'args': ((output_paths['codah']['statement']['train'],), output_paths['codah']['statement']['vocab'])},
            {'func': ground, 'args': (output_paths['codah']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['codah']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['codah']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['codah']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['codah']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['codah']['grounded']['test'], args.nprocs)},
            {'func': find_paths, 'args': (output_paths['codah']['grounded']['train'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['codah']['paths']['raw-train'], args.nprocs, args.seed)},
            {'func': find_paths, 'args': (output_paths['codah']['grounded']['dev'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['codah']['paths']['raw-dev'], args.nprocs, args.seed)},
            {'func': find_paths, 'args': (output_paths['codah']['grounded']['test'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['codah']['paths']['raw-test'], args.nprocs, args.seed)},
            {'func': score_paths, 'args': (output_paths['codah']['paths']['raw-train'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['codah']['paths']['scores-train'], args.nprocs)},
            {'func': score_paths, 'args': (output_paths['codah']['paths']['raw-dev'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['codah']['paths']['scores-dev'], args.nprocs)},
            {'func': score_paths, 'args': (output_paths['codah']['paths']['raw-test'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['codah']['paths']['scores-test'], args.nprocs)},
            {'func': prune_paths, 'args': (output_paths['codah']['paths']['raw-train'], output_paths['codah']['paths']['scores-train'],
                                           output_paths['codah']['paths']['pruned-train'], args.path_prune_threshold)},
            {'func': prune_paths, 'args': (output_paths['codah']['paths']['raw-dev'], output_paths['codah']['paths']['scores-dev'],
                                           output_paths['codah']['paths']['pruned-dev'], args.path_prune_threshold)},
            {'func': prune_paths, 'args': (output_paths['codah']['paths']['raw-test'], output_paths['codah']['paths']['scores-test'],
                                           output_paths['codah']['paths']['pruned-test'], args.path_prune_threshold)},
            {'func': generate_graph, 'args': (output_paths['codah']['grounded']['train'], output_paths['codah']['paths']['pruned-train'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['codah']['graph']['train'])},
            {'func': generate_graph, 'args': (output_paths['codah']['grounded']['dev'], output_paths['codah']['paths']['pruned-dev'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['codah']['graph']['dev'])},
            {'func': generate_graph, 'args': (output_paths['codah']['grounded']['test'], output_paths['codah']['paths']['pruned-test'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['codah']['graph']['test'])},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['codah']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['codah']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['codah']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['codah']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['codah']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['codah']['graph']['adj-test'], args.nprocs)},
            {'func': generate_triples_from_adj, 'args': (output_paths['codah']['graph']['adj-train'], output_paths['codah']['grounded']['train'],
                                                         output_paths['cpnet']['vocab'], output_paths['codah']['triple']['train'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['codah']['graph']['adj-dev'], output_paths['codah']['grounded']['dev'],
                                                         output_paths['cpnet']['vocab'], output_paths['codah']['triple']['dev'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['codah']['graph']['adj-test'], output_paths['codah']['grounded']['test'],
                                                         output_paths['cpnet']['vocab'], output_paths['codah']['triple']['test'])},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['codah']['graph']['adj-train'], output_paths['cpnet']['pruned-graph'], output_paths['codah']['paths']['adj-train'], output_paths['codah']['graph']['nxg-from-adj-train'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['codah']['graph']['adj-dev'], output_paths['cpnet']['pruned-graph'], output_paths['codah']['paths']['adj-dev'], output_paths['codah']['graph']['nxg-from-adj-dev'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['codah']['graph']['adj-test'], output_paths['cpnet']['pruned-graph'], output_paths['codah']['paths']['adj-test'], output_paths['codah']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
