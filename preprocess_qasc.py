import os
import argparse
from multiprocessing import cpu_count
from utils.build_qasc import build_qasc
from utils.convert_qasc import convert_to_entailment
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from utils.conceptnet import extract_english, construct_graph
from utils.embedding import glove2npy, load_pretrained_embeddings
from utils.grounding import create_matcher_patterns, ground
from utils.paths import find_paths, score_paths, prune_paths, find_relational_paths_from_paths, generate_path_and_graph_from_adj
from utils.graph import generate_graph, generate_adj_data_from_grounded_concepts, coo_to_normalized
from utils.triples import generate_triples_from_adj

input_paths = {
    'qasc': {
        'train': './data/qasc/train.jsonl',
        'train_2step': './data/qasc/2step/train_2step_raw.jsonl',
        'dev': './data/qasc/dev.jsonl',
        'dev_2step': './data/qasc/2step/dev_2step_raw.jsonl',
        'test': './data/qasc/test.jsonl',
        'test_2step': './data/qasc/2step/test_2step_raw.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'glove': {
        'txt': './data/glove/glove.6B.300d.txt',
    },
    'numberbatch': {
        'txt': './data/transe/numberbatch-en-19.08.txt',
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
    'qasc': {
        'dataset': {
            'train': './data/qasc/train_2step.jsonl',
            'dev': './data/qasc/dev_2step.jsonl',
            'test': './data/qasc/test_2step.jsonl',
        },
        'statement': {
            'train': './data/qasc/statement/train.statement.jsonl',
            'dev': './data/qasc/statement/dev.statement.jsonl',
            'test': './data/qasc/statement/test.statement.jsonl',
            'vocab': './data/qasc/statement/vocab.json',
        },
        'statement-with-ans-pos': {
            'train': './data/qasc/statement/train.statement-with-ans-pos.jsonl',
            'dev': './data/qasc/statement/dev.statement-with-ans-pos.jsonl',
            'test': './data/qasc/statement/test.statement-with-ans-pos.jsonl',
        },
        'tokenized': {
            'train': './data/qasc/tokenized/train.tokenized.txt',
            'dev': './data/qasc/tokenized/dev.tokenized.txt',
            'test': './data/qasc/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/qasc/grounded/train.grounded.jsonl',
            'dev': './data/qasc/grounded/dev.grounded.jsonl',
            'test': './data/qasc/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': './data/qasc/paths/train.paths.raw.jsonl',
            'raw-dev': './data/qasc/paths/dev.paths.raw.jsonl',
            'raw-test': './data/qasc/paths/test.paths.raw.jsonl',
            'scores-train': './data/qasc/paths/train.paths.scores.jsonl',
            'scores-dev': './data/qasc/paths/dev.paths.scores.jsonl',
            'scores-test': './data/qasc/paths/test.paths.scores.jsonl',
            'pruned-train': './data/qasc/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/qasc/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/qasc/paths/test.paths.pruned.jsonl',
            'adj-train': './data/qasc/paths/train.paths.adj.jsonl',
            'adj-dev': './data/qasc/paths/dev.paths.adj.jsonl',
            'adj-test': './data/qasc/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/qasc/graph/train.graph.jsonl',
            'dev': './data/qasc/graph/dev.graph.jsonl',
            'test': './data/qasc/graph/test.graph.jsonl',
            'adj-train': './data/qasc/graph/train.graph.adj.pk',
            'adj-dev': './data/qasc/graph/dev.graph.adj.pk',
            'adj-test': './data/qasc/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/qasc/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/qasc/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/qasc/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/qasc/triples/train.triples.pk',
            'dev': './data/qasc/triples/dev.triples.pk',
            'test': './data/qasc/triples/test.triples.pk',
        }
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['qasc'], choices=['qasc'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'qasc': [
            {'func': build_qasc, 'args': (input_paths['qasc']['train'], input_paths['qasc']['train_2step'], output_paths['qasc']['dataset']['train'])},
            {'func': build_qasc, 'args': (input_paths['qasc']['dev'], input_paths['qasc']['dev_2step'], output_paths['qasc']['dataset']['dev'])},
            {'func': build_qasc, 'args': (input_paths['qasc']['test'], input_paths['qasc']['test_2step'], output_paths['qasc']['dataset']['test'])},
            {'func': convert_to_entailment, 'args': (input_paths['qasc']['train'], output_paths['qasc']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['qasc']['dev'], output_paths['qasc']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['qasc']['test'], output_paths['qasc']['statement']['test'])},
            {'func': tokenize_statement_file, 'args': (output_paths['qasc']['statement']['train'], output_paths['qasc']['tokenized']['train'])},
            {'func': tokenize_statement_file, 'args': (output_paths['qasc']['statement']['dev'], output_paths['qasc']['tokenized']['dev'])},
            {'func': tokenize_statement_file, 'args': (output_paths['qasc']['statement']['test'], output_paths['qasc']['tokenized']['test'])},
            {'func': make_word_vocab, 'args': ((output_paths['qasc']['statement']['train'],), output_paths['qasc']['statement']['vocab'])},
            {'func': ground, 'args': (output_paths['qasc']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['qasc']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['qasc']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['qasc']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['qasc']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['qasc']['grounded']['test'], args.nprocs)},
            {'func': find_paths, 'args': (output_paths['qasc']['grounded']['train'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['qasc']['paths']['raw-train'], args.nprocs, args.seed)},
            {'func': find_paths, 'args': (output_paths['qasc']['grounded']['dev'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['qasc']['paths']['raw-dev'], args.nprocs, args.seed)},
            {'func': find_paths, 'args': (output_paths['qasc']['grounded']['test'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['qasc']['paths']['raw-test'], args.nprocs, args.seed)},
            {'func': score_paths, 'args': (output_paths['qasc']['paths']['raw-train'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['qasc']['paths']['scores-train'], args.nprocs)},
            {'func': score_paths, 'args': (output_paths['qasc']['paths']['raw-dev'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['qasc']['paths']['scores-dev'], args.nprocs)},
            {'func': score_paths, 'args': (output_paths['qasc']['paths']['raw-test'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['qasc']['paths']['scores-test'], args.nprocs)},
            {'func': prune_paths, 'args': (output_paths['qasc']['paths']['raw-train'], output_paths['qasc']['paths']['scores-train'],
                                           output_paths['qasc']['paths']['pruned-train'], args.path_prune_threshold)},
            {'func': prune_paths, 'args': (output_paths['qasc']['paths']['raw-dev'], output_paths['qasc']['paths']['scores-dev'],
                                           output_paths['qasc']['paths']['pruned-dev'], args.path_prune_threshold)},
            {'func': prune_paths, 'args': (output_paths['qasc']['paths']['raw-test'], output_paths['qasc']['paths']['scores-test'],
                                           output_paths['qasc']['paths']['pruned-test'], args.path_prune_threshold)},
            {'func': generate_graph, 'args': (output_paths['qasc']['grounded']['train'], output_paths['qasc']['paths']['pruned-train'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['qasc']['graph']['train'])},
            {'func': generate_graph, 'args': (output_paths['qasc']['grounded']['dev'], output_paths['qasc']['paths']['pruned-dev'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['qasc']['graph']['dev'])},
            {'func': generate_graph, 'args': (output_paths['qasc']['grounded']['test'], output_paths['qasc']['paths']['pruned-test'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['qasc']['graph']['test'])},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['qasc']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['qasc']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['qasc']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['qasc']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['qasc']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['qasc']['graph']['adj-test'], args.nprocs)},
            {'func': generate_triples_from_adj, 'args': (output_paths['qasc']['graph']['adj-train'], output_paths['qasc']['grounded']['train'],
                                                         output_paths['cpnet']['vocab'], output_paths['qasc']['triple']['train'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['qasc']['graph']['adj-dev'], output_paths['qasc']['grounded']['dev'],
                                                         output_paths['cpnet']['vocab'], output_paths['qasc']['triple']['dev'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['qasc']['graph']['adj-test'], output_paths['qasc']['grounded']['test'],
                                                         output_paths['cpnet']['vocab'], output_paths['qasc']['triple']['test'])},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['qasc']['graph']['adj-train'], output_paths['cpnet']['pruned-graph'], output_paths['qasc']['paths']['adj-train'], output_paths['qasc']['graph']['nxg-from-adj-train'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['qasc']['graph']['adj-dev'], output_paths['cpnet']['pruned-graph'], output_paths['qasc']['paths']['adj-dev'], output_paths['qasc']['graph']['nxg-from-adj-dev'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['qasc']['graph']['adj-test'], output_paths['cpnet']['pruned-graph'], output_paths['qasc']['paths']['adj-test'], output_paths['qasc']['graph']['nxg-from-adj-test'], args.nprocs)},
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()