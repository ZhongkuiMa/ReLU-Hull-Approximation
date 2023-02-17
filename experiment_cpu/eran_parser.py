"""
This is our revised parser from ERAN with only needed arguments.
"""
import argparse
import os

from experiment_cpu.eran_args import ERANArgs

SUPPORTED_DOMAINS = ("deeppoly", "refinepoly", "gpupoly", "refinegpupoly")
SUPPORTED_FILE_EXTENSIONS = (".onnx",)
SUPPORTED_DATASETS = ("mnist", "cifar10")

SUPPORTED_CONVEX_METHODS = ("cdd", "fast", "sci", "sciplus")


class ERANParser(argparse.ArgumentParser):
    def __init__(self, args: ERANArgs):
        argparse.ArgumentParser.__init__(self, description="ERAN Example",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # About basic information
        self.add_argument("--net_file", type=is_network_file, default=args.net_file,
                          help="The network file name/path (.pb, .pyt, .tf, .meta, or .onnx)")
        self.add_argument("--dataset", type=is_dataset, default=args.dataset,
                          help="The dataset (mnist, cifar10, acasxu, or fashion)")
        self.add_argument("--domain", type=is_domain, default=args.domain,
                          help="The domain name (deeppoly, refinepoly).")
        self.add_argument("--epsilon", type=float, default=args.epsilon,
                          help="The Epsilon for L_infinity perturbation")
        # About samples number
        self.add_argument("--samples_num", type=is_positive_int, default=args.samples_num,
                          help="The number of samples to test")
        self.add_argument("--samples_start", type=int, default=args.samples_start,
                          help="The first id of samples to test")

        # About k-activation refinement
        self.add_argument("--ns", type=is_positive_int, default=args.ns,
                          help="The number of variables to group by k-activation")
        self.add_argument("--k", type=is_positive_int, default=args.k,
                          help='The group size of k-activation')
        self.add_argument("--s", type=is_positive_int, default=args.s,
                          help='The overlap size between two k-activation group')
        self.add_argument("--use_default_heuristic", action="store_true", default=args.use_heuristic,
                          help="Whether to use the area heuristic for the k-activation approximation "
                               "or to always create new noise symbols per relu for the DeepZono ReLU approximation")
        self.add_argument("--convex_method", type=is_convex_method, default=args.convex_method,
                          help="The method to calculate k-activation")
        self.add_argument("--use_cutoff_of", type=float, default=args.use_cutoff_of,
                          help="Used to ignore some groups when encoding k activation; otherwise, the final LP"
                               "verifying problem sometimes maybe infeasible.")
        # About timeout
        self.add_argument("--timeout_lp", type=is_positive_float, default=args.timeout_lp,
                          help="The timeout for the LP solver to refine")
        self.add_argument("--timeout_final_lp", type=is_positive_float, default=args.timeout_final_lp,
                          help="The timeout for the final LP solver to final verify")
        self.add_argument("--timeout_milp", type=is_positive_float, default=args.timeout_milp,
                          help="The timeout for the MILP solver to refine")
        self.add_argument("--timeout_final_milp", type=is_positive_float, default=args.timeout_final_lp,
                          help="The timeout for the final MILP solve to final verify")
        # About general settings
        self.add_argument("--processes_num", type=int, default=args.processes_num,
                          help="The number of processes for MILP/LP/k-activation")

    def set_args(self, args: ERANArgs):
        arguments = self.parse_args()
        for k, v in vars(arguments).items():
            setattr(args, k, v)
        args.json = vars(arguments)


def is_domain(domain: str) -> str:
    if domain not in SUPPORTED_DOMAINS:
        raise argparse.ArgumentTypeError(f"{domain} is not supported. Only support {SUPPORTED_DOMAINS}.")
    return domain


def is_dataset(dataset: str) -> str:
    if dataset not in SUPPORTED_DATASETS:
        raise argparse.ArgumentTypeError(f"{dataset} is not supported. Only support {SUPPORTED_DATASETS}.")
    return dataset


def is_network_file(net_file: str) -> str:
    if not os.path.isfile(net_file):
        raise argparse.ArgumentTypeError(f'The net file "{net_file}" is not found.')
    _, extension = os.path.splitext(net_file)
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise argparse.ArgumentTypeError(f"{extension} is not supported. Only support {SUPPORTED_FILE_EXTENSIONS}.")
    return net_file


def is_convex_method(method: str) -> str:
    if method not in SUPPORTED_CONVEX_METHODS:
        raise argparse.ArgumentTypeError(f"{method} is not supported. Only support {SUPPORTED_CONVEX_METHODS}.")
    return method


def is_positive_int(number: str) -> int:
    try:
        number = int(number)
    except Exception as e:
        raise argparse.ArgumentTypeError("This argument should be an integer.") from e
    if number <= 0:
        raise argparse.ArgumentTypeError("This argument should be positive integer.")
    return number


def is_positive_float(number: str) -> float:
    try:
        number = float(number)
    except Exception as e:
        raise argparse.ArgumentTypeError("This argument should be a float number.") from e

    if number <= 0:
        raise argparse.ArgumentTypeError("This argument should be positive float number.")
    return number
