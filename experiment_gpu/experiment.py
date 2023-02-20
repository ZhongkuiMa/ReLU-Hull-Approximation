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
from config import config
from tf_verify_sci.read_net_file import read_onnx_net
from tf_verify_sci.analyzer import layers
from tf_verify_sci.onnx_translator import ONNXTranslator
from tf_verify_sci.optimizer import Optimizer
from tf_verify_sci.refine_gpupoly import refine_gpupoly_results
from experiment_cpu.experiment import initialise_args, parse_net_name, initialise_samples_data, skip_sample, \
    stop_experiment, get_inputs
from experiment_cpu.samples_incorrect_classfied import INCORRECTLY_CLASSIFIED
from experiment_cpu.samples_verified_by_dp import VERIFIED_SAMPLES_BY_DEEPPOLY
from experiment_cpu.utils import normalize

IGNORE_SAMPLES_VERIFIED_BY_DEEPPOLY = False

def initialise_gpupoly_model(net_file_path):
    model, is_conv = read_onnx_net(net_file_path)
    translator = ONNXTranslator(model, True)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    nn = layers()
    network, relu_layer_ids, gpu_layers_num = optimizer.get_gpupoly(nn)

    return network, nn, relu_layer_ids, gpu_layers_num, is_conv


def run_experiment():
    args = initialise_args()

    domain = config.domain = args.domain
    dataset = config.dataset = args.dataset
    net_file_path = config.netname = args.net_file
    config.from_test = args.samples_start
    config.num_tests = args.samples_num
    print(config.from_test, config.num_tests)
    krelu_method = config.approx_k = args.convex_method
    epsilon = config.epsilon = args.epsilon
    ns = config.sparse_n = args.ns
    k = config.k = args.k
    s = config.s = args.s

    config.from_test = args.samples_start
    config.num_tests = args.samples_num

    net_name = parse_net_name(net_file_path)
    network, nn, relu_layer_ids, gpu_layers_num, is_conv = initialise_gpupoly_model(net_file_path)
    samples, means, stds = initialise_samples_data(dataset)

    correctly_classified_num = 0
    verified_num = 0
    total_time = 0
    verified_by_deeppoly = 0

    ignored_samples = []
    if IGNORE_SAMPLES_VERIFIED_BY_DEEPPOLY:
        ignored_samples = INCORRECTLY_CLASSIFIED[net_name]
        if domain == "refinegpupoly":
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

        is_correctly_classified = network.test(specLB, specUB, label)
        if not is_correctly_classified:
            print("[RESULT] Incorrectly classified. ")
            samples_ic.append(i)
            continue
        print("This sample is correctly classified.")

        print("Verify the sample by GPUPoly...")
        start = time.time()
        correctly_classified_num += 1

        specLB = np.clip(image.copy() - epsilon, 0, 1)
        specUB = np.clip(image.copy() + epsilon, 0, 1)
        normalize(specLB, means, stds, dataset, domain, is_conv=is_conv)
        normalize(specUB, means, stds, dataset, domain, is_conv=is_conv)

        if network.test(specLB, specUB, label):
            print("[RESULT] Verified by GPUPoly.")
            samples_dp.append(i)
            verified_by_deeppoly += 1
            verified_num += 1

        elif domain == "refinegpupoly":
            max_try_times = 2 if krelu_method != "fast" else 0
            try_times = 0
            print("Verify the sample by RefineGPUPoly...")
            while True:
                ouputs_num = len(nn.weights[-1])

                # Matrix that computes the difference with the expected layer.
                diff_matrix = np.delete(-np.eye(ouputs_num), label, 0)
                diff_matrix[:, label] = 1
                diff_matrix = diff_matrix.astype(np.float64)
                # gets the values from GPUPoly.
                res = network.evalAffineExpr(diff_matrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)

                labels_to_be_verified = []
                nn.specLB, nn.specUB = specLB, specUB
                nn.predecessors = []
                for pred in range(nn.numlayer + 1):
                    predecessor = np.zeros(1, dtype=np.int)
                    predecessor[0] = int(pred - 1)
                    nn.predecessors.append(predecessor)

                var = 0
                for l in range(ouputs_num):
                    if l != label:
                        if res[var][0] < 0:
                            labels_to_be_verified.append(l)
                        var += 1

                is_verified_by_gpupoly, x, model_success = refine_gpupoly_results(
                    nn, network, gpu_layers_num, relu_layer_ids, label, labels_to_be_verified, K=k, s=s,
                    method=krelu_method)

                if not model_success and try_times < max_try_times:
                    config.cutoff += 0.02
                    try_times+=1
                    continue

                if is_verified_by_gpupoly:
                    print("[RESULT] Verified by RefineGPUPoly.")
                    samples_rp.append(i)
                    verified_num += 1
                    break
                print("[RESULT] Fialed to verify by RefineGPUPoly.")
                break
        else:
            print("[RESULT] Fialed to verify by GPUPoly.")

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

        if (domain == "gpupoly" and correctly_classified_num >= 100) \
                or (domain == "refinegpupoly" and correctly_classified_num + len(
            VERIFIED_SAMPLES_BY_DEEPPOLY[net_name]) >= 100):
            break

if __name__ == '__main__':
    run_experiment()