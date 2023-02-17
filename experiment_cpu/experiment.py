import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.sched_setaffinity(0, os.sched_getaffinity(0))
import sys

sys.path.insert(0, '../')
sys.path.insert(0, '../tf_verify_sci/')
sys.path.insert(0, '../../ELINA/python_interface/')

import time
from datetime import datetime
import numpy as np
from tf_verify_sci.config import config
from tf_verify_sci.read_net_file import read_onnx_net
from tf_verify_sci.eran import ERAN
from experiment_cpu.eran_args import ERANArgs
from experiment_cpu.eran_parser import ERANParser
from experiment_cpu.samples_verified_by_dp import VERIFIED_SAMPLES_BY_DEEPPOLY
from experiment_cpu.samples_incorrect_classfied import INCORRECTLY_CLASSIFIED
from experiment_cpu.utils import get_means_stds, get_samples, normalize

IGNORE_SAMPLES_VERIFIED_BY_DEEPPOLY = False


def initialise_args():
    args = ERANArgs()
    parser = ERANParser(args)
    parser.set_args(args)
    args.prepare_and_check()
    args.print_args()

    return args


def initialise_deeppoly_model(net_file_path):
    model, is_conv = read_onnx_net(net_file_path)
    eran = ERAN(model, is_onnx=True)

    return eran, is_conv


def initialise_samples_data(dataset):
    means, stds = get_means_stds(dataset)
    samples = get_samples(dataset)
    return samples, means, stds


def parse_net_name(net_file_path):
    net_file_path = net_file_path.split('/')[-1].split('.')
    if len(net_file_path) > 2:
        net_file_path = ".".join(net_file_path[i] for i in range(len(net_file_path) - 1))
    else:
        net_file_path = net_file_path[0]
    return net_file_path


def skip_sample(i, ignored_samples):
    return (config.domain == "refinepoly" and i in ignored_samples) \
           or (config.domain == "refinegpupoly" and i in ignored_samples) \
           or (config.from_test is not None and i < config.from_test)


def stop_experiment(i):
    return config.num_tests is not None and i >= config.from_test + config.num_tests


def get_inputs(test, means, stds, dataset, domain, is_conv):
    image = np.float64(test[1:]) / np.float64(255)
    specLB, specUB = np.copy(image), np.copy(image)
    normalize(specLB, means, stds, dataset, domain, is_conv)
    normalize(specUB, means, stds, dataset, domain, is_conv)
    label = int(test[0])
    return image, specLB, specUB, label


def run_experiment():
    args = initialise_args()

    domain = config.domain = args.domain
    dataset = config.dataset = args.dataset
    net_file_path = config.netname = args.net_file
    krelu_method = config.approx_k = args.convex_method

    epsilon = config.epsilon = args.epsilon
    ns = config.sparse_n = args.ns
    k = config.k = args.k
    s = config.s = args.s

    net_name = parse_net_name(net_file_path)
    eran, is_conv = initialise_deeppoly_model(net_file_path)
    samples, means, stds = initialise_samples_data(dataset)

    correctly_classified_num = 0
    verified_num = 0
    total_time = 0
    verified_by_deeppoly = 0

    ignored_samples = INCORRECTLY_CLASSIFIED[net_name]
    if domain == "refinepoly":
        ignored_samples = ignored_samples + VERIFIED_SAMPLES_BY_DEEPPOLY[net_name]

    samples_ic = []
    samples_dp = []
    samples_rp = []

    created_time = datetime.now()

    for i, test in enumerate(samples):
        if skip_sample(i, ignored_samples):
            continue
        if stop_experiment(i):
            break

        print(f"SAMPLE-{i}, {domain}, {net_name}".center(100, "-"))
        print(f"[SETTINGS] epsilon={epsilon}, ns={ns}, k={k}, s={s}, krelu_method={krelu_method}")
        print("Check whether the sample is correctly classified...")
        image, specLB, specUB, label = get_inputs(test, means, stds, dataset, domain, is_conv)

        label, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, "deeppoly")
        if label != int(test[0]):
            print("[RESULT] Incorrectly classified. ")
            samples_ic.append(i)
            continue
        print("This sample is correctly classified.")

        print("Verify the sample by DeepPoly...")
        start = time.time()
        correctly_classified_num += 1

        specLB = np.clip(image.copy() - epsilon, 0, 1)
        specUB = np.clip(image.copy() + epsilon, 0, 1)
        normalize(specLB, means, stds, dataset, domain, is_conv=is_conv)
        normalize(specUB, means, stds, dataset, domain, is_conv=is_conv)

        results = eran.analyze_box(specLB, specUB, "deeppoly", label=label, K=0, s=0, approx_k=0)
        perturbed_label, _, nlb, nub, failed_labels, x = results
        if perturbed_label == label:
            print("[RESULT] Verified by DeepPoly.")
            samples_dp.append(i)
            verified_by_deeppoly += 1
            verified_num += 1

        elif domain == "refinepoly":
            print("Verify the sample by RefinePoly...")
            results = eran.analyze_box(specLB, specUB, domain, label=label)
            perturbed_label, _, nlb, nub, failed_labels, x = results
            if perturbed_label == label:
                print("[RESULT] Verified by RefinePoly.")
                samples_rp.append(i)
                verified_num += 1
            else:
                print("[RESULT] Fialed to verify by RefinePoly.")

        else:
            print("[RESULT] Fialed to verify by DeepPoly.")

        used_time = time.time() - start
        total_time += used_time
        print(f"[STATS]\n"
              f"epsilon={epsilon}, krelu_method={krelu_method}, ns={ns}, k={k}, s={s}\n"
              f"[CORRECTED]:    {correctly_classified_num}\n"
              f"[VERIFIED DP]:  {verified_by_deeppoly}\n"
              f"[VERIFIED]:     {verified_num} / {correctly_classified_num}\n"
              f"[INCORRECTED]:  {samples_ic}\n"
              f"[VERIFIED DP]:  {samples_dp}\n"
              f"[VERIFIED RP]:  {samples_rp}\n"
              f"[USED TIME]:    {used_time:.4f}s\n"
              f"[AVERAGE TIME]: {0 if total_time == 0 else total_time / correctly_classified_num:.4f}s\n"
              f"[TOTAL TIME]:   {total_time:.4f}s\n")

        fp = f"./logs/{dataset}/{net_name}_{domain}_{epsilon}_{krelu_method}_{ns}_{k}_{s}_{created_time}.txt"
        mode = "a" if os.path.exists(fp) else "w"
        with open(fp, mode) as file:
            file.write(f"sample-{i}, {dataset}, {net_name}, {domain}".center(100, "-") +
                       f"\nepsilon={epsilon}, krelu_method={krelu_method}, ns={ns}, k={k}, s={s}\n"
                       f"[CORRECTED]:    {correctly_classified_num}\n"
                       f"[VERIFIED DP]:  {verified_by_deeppoly}\n"
                       f"[VERIFIED]:     {verified_num}/{correctly_classified_num}\n"
                       f"[INCORRECTED]:  {samples_ic}\n"
                       f"[VERIFIED DP]:  {samples_dp}\n"
                       f"[VERIFIED RP]:  {samples_rp}\n"
                       f"[USED TIME]:    {used_time:.4f}s\n"
                       f"[AVERAGE TIME]: {0 if total_time == 0 else total_time / correctly_classified_num:.4f}s\n"
                       f"[TOTAL TIME]:   {total_time:.4f}s\n")

        if (domain == "deeppoly" and correctly_classified_num >= 100) \
                or (domain == "refinepoly" and correctly_classified_num + len(
            VERIFIED_SAMPLES_BY_DEEPPOLY[net_name]) >= 100):
            break
