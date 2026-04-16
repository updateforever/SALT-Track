import argparse

import _init_paths
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze tracking results.')
    parser.add_argument('--tracker_name', type=str, default='atctrack',
                        help='tracker name, e.g. atctrack')
    parser.add_argument('--tracker_param', type=str, required=True,
                        help='tracker parameter/config name, e.g. atctrack_base_semantic_visual_only_h100_short')
    parser.add_argument('--dataset_name', type=str, default='tnl2k',
                        help='dataset name, e.g. tnl2k, lasot, otb99_lang')
    parser.add_argument('--run_ids', type=int, nargs='*', default=None,
                        help='optional run ids for repeated runs')
    parser.add_argument('--display_name', type=str, default=None,
                        help='optional tracker display name in result tables')
    parser.add_argument('--no_merge', action='store_true',
                        help='disable merge_results for repeated runs')
    parser.add_argument('--force_evaluation', action='store_true',
                        help='recompute cached evaluation data')
    return parser.parse_args()


def main():
    args = parse_args()

    trackers = trackerlist(name=args.tracker_name,
                           parameter_name=args.tracker_param,
                           dataset_name=args.dataset_name,
                           run_ids=args.run_ids,
                           display_name=args.display_name)

    dataset = get_dataset(args.dataset_name)

    print_results(trackers,
                  dataset,
                  args.dataset_name,
                  merge_results=not args.no_merge,
                  plot_types=('success', 'prec', 'norm_prec'),
                  force_evaluation=args.force_evaluation)


if __name__ == '__main__':
    main()
