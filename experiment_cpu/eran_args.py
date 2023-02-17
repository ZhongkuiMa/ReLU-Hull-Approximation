"""
This is our revised arguments of ERAN with only needed ones.
"""
import multiprocessing
from enum import Enum


class Device(Enum):
    CPU = 0
    CUDA = 1


class ERANArgs:
    def __init__(self):
        # About basic information
        self.net_file = ""
        self.dataset = ""
        self.domain = ""
        self.epsilon = 0.0

        # About samples number
        self.samples_num = None
        self.samples_start = 0

        # About k-activation refinement
        self.ns = 100
        self.k = 3
        self.s = self.k - 1
        self.use_heuristic = True
        self.convex_method = "fast"
        self.use_cutoff_of = 0.05

        self.timeout_lp = 10
        self.timeout_milp = 10
        self.timeout_final_lp = 100
        self.timeout_final_milp = 100

        self.processes_num = multiprocessing.cpu_count()

        # GPU options
        self.device = Device.CPU

        self.means = None
        self.stds = None

    def prepare_and_check(self):
        if self.dataset == "cifar10":
            self.means, self.stds = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        else:
            self.means, self.stds = [0.0], [1.0]

    def print_args(self):
        print("Model Arguments".center(100, "-"))
        print(f"dataset={self.dataset}, net_file={self.net_file}")
        print(f"samples_start={self.samples_start}, samples_num={self.samples_num}")
        print(f"domain={self.domain}")
        print(f"epsilon={self.epsilon}")
        print(f"convex_method={self.convex_method}")
        print(f"ns={self.ns}, k={self.k}, s={self.s}")
        print()
