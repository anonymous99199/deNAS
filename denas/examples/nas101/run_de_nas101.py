import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks_development/'))

import json
import pickle
import argparse
import numpy as np

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

from denas import DE


def remove_invalid_configs(traj, runtime, history):
    idx = np.where(np.array(runtime)==0)
    runtime = np.delete(runtime, idx)
    traj = np.delete(np.array(traj), idx)
    history = np.delete(history, idx, axis=0)
    return traj, runtime, history

def calc_test(history):
    global b, y_star_valid, y_star_test, inc_config
    de_dummy = DE(cs=cs)
    regret_test = []
    inc = np.inf
    for i in range(len(history)):
        config = history[i][0]
        config = de_dummy.vector_to_configspace(config)
        test_error, _ = b.objective_function_test(config)
        if test_error < inc:
            inc = test_error
        regret_test.append(inc)
    return regret_test

def save(trajectory, runtime, history, output_path, run_id, filename="run"):
    global b, y_star_valid, y_star_test, inc_config
    trajectory = trajectory - y_star_valid
    res = {}
    res["runtime"] = np.cumsum(runtime).tolist()
    if np.max(trajectory) < 0:
        a_min = -np.inf
        a_max = 0
    else:
        a_min = 0
        a_max = np.inf
    res["regret_validation"] = np.array(np.clip(trajectory, a_min=a_min, a_max=a_max)).tolist()
    res["history"] = history.tolist()
    res['y_star_valid'] = float(y_star_valid)
    res['y_star_test'] = float(y_star_test)
    res['inc_config'] = inc_config
    if 'cifar' in args.benchmark:
        res_new = b.get_results(ignore_invalid_configs=True)
        res['regret_test'] = res_new['regret_test']
    else:
        test_regret = np.array(calc_test(res["history"])) - y_star_test
        res['regret_test'] = np.array(np.clip(test_regret, a_min=a_min, a_max=a_max)).tolist()
    fh = open(os.path.join(output_path, '{}_{}.json'.format(filename, run_id)), 'w')
    json.dump(res, fh)
    fh.close()

def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()

def f(config, budget=None):
    if budget is not None:
        fitness, cost = b.objective_function(config, budget=int(budget))
    else:
        fitness, cost = b.objective_function(config)
    return fitness, cost

parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
choices = ["protein_structure", "slice_localization", "naval_propulsion",
           "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b", "nas_cifar10c"]
parser.add_argument('--benchmark', default="protein_structure", type=str,
                    help="specify the benchmark to run on from among {}".format(choices))
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=10, type=int, nargs='?', help='population size')
strategy_choices = ['rand1_bin', 'rand2_bin', 'rand2dir_bin', 'best1_bin', 'best2_bin',
                    'currenttobest1_bin', 'randtobest1_bin',
                    'rand1_exp', 'rand2_exp', 'rand2dir_exp', 'best1_exp', 'best2_exp',
                    'currenttobest1_exp', 'randtobest1_exp']
parser.add_argument('--strategy', default="rand1_bin", choices=strategy_choices,
                    type=str, nargs='?',
                    help="specify the DE strategy from among {}".format(strategy_choices))
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                    help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                    help='probability of crossover')
parser.add_argument('--max_budget', default=None, type=str, nargs='?',
                    help='maximum wallclock time to run DE for')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='de', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.benchmark == "nas_cifar10a":
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10b":
    min_budget = 4
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10c":
    min_budget = 4
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "protein_structure":
    min_budget = 4
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "slice_localization":
    min_budget = 4
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "naval_propulsion":
    min_budget = 4
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring":
    min_budget = 4
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()


cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.folder)
os.makedirs(output_path, exist_ok=True)

# Initializing DE object
de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
        mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
        strategy=args.strategy, max_budget=args.max_budget)

if args.runs is None:
    if not args.fix_seed:
        np.random.seed(0)
    # Running DE iterations
    traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
    if 'cifar' in args.benchmark:
        res = b.get_results(ignore_invalid_configs=True)
    else:
        res = b.get_results()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()
    # if 'cifar' in args.benchmark:
    #     # save(traj, runtime, history, output_path, args.run_id, filename="raw_run")
    #     traj, runtime, history = remove_invalid_configs(traj, runtime, history)
    # save(traj, runtime, history, output_path, args.run_id)
else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DE iterations
        traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
        if 'cifar' in args.benchmark:
            res = b.get_results(ignore_invalid_configs=True)
        else:
            res = b.get_results()
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
        # if 'cifar' in args.benchmark:
        #     # save(traj, runtime, history, output_path, run_id, filename="raw_run")
        #     traj, runtime, history = remove_invalid_configs(traj, runtime, history)
        # save(traj, runtime, history, output_path, run_id)
        print("Run saved. Resetting...")
        de.reset()
        b.reset_tracker()

save_configspace(cs, output_path)
