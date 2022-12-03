import argparse, sys, os
from copy import deepcopy
from Common.Utils import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

parser = argparse.ArgumentParser(description='Results integrated plot')

parser.add_argument('--base_path', default="/home/phb/ETRI/GymSim_s2r_2/", help='base path of the current project')
parser.add_argument("--env_name", "-en", default="Hopper-v4", type=str, help="the name of environment to show")

args = parser.parse_known_args()

if len(args) != 1:
    args = args[0]

def get_env_file(results_list):
    specific_file = []

    for result_name in results_list:
        if args.env_name in result_name:
            specific_file.append(result_name)

    return specific_file

def find_each_results(results_list):
    results_npy = {}

    results_npy["reward"] = {}
    results_npy["success_rate"] = {}
    results_npy["model_error"] = {}

    for results in results_list:
        result_path = "./results/" + results + "/log/test/"
        test_log_list = os.listdir(result_path)
        for test_log in test_log_list:
            if "npy" in test_log:
                terms = test_log[:-4].split("_")

                indicator_name = terms[0] if "reward" in terms else "_".join(terms[:2])
                type_name = "none" if "none" in terms else terms[-2]
                case_name = terms[-1]

                results_npy[indicator_name][case_name] = results_npy[indicator_name].get(case_name, {})
                results_npy[indicator_name][case_name][type_name] = np.load(result_path + test_log)

    return results_npy

def get_color(net_type):
    if net_type == "bnn":
        return "red"
    elif net_type == "dnn":
        return "blue"
    else:
        return "black"

def plot_variance(plt, data, color, label):

    x = range(len(data[:, 0]))
    plt.plot(x, data[:, 0], color=color, label=label)
    y1 = np.asarray(data[:, 0]) + np.asarray(data[:, 1])
    y2 = np.asarray(data[:, 0]) - np.asarray(data[:, 1])
    plt.fill_between(x, y1, y2, alpha=0.3, color=color)

def get_rewards_plot(plt, data):

    for net_type in data.keys():
        plot_variance(plt, data[net_type], color=get_color(net_type), label=net_type)

def get_success_rate_plot(plt, data):
    global num_data

    width = 0.2
    num_net_type = len(data.keys())
    for i, net_type in enumerate(data.keys()):

        x = np.arange(num_data)  # the label locations

        x_new = x + (i-(num_net_type-1)/2)*width
        rects = plt.bar(x_new, data[net_type].flatten(), width, color=get_color(net_type), label=net_type)
        # plt.bar_label(rects, padding=3)

    # plt.tight_layout()

def get_selected_data(original_data):

    new_data = deepcopy(original_data)
    new = new_data["reward"]["disturb"]["dnn"][:, 0]
    new_data["reward"]["disturb"]["dnn"][:, 0] = new[new % 2 == 0]

    return new_data

def main():
    global num_data

    results_list = get_env_file(os.listdir("./results"))
    result_data = find_each_results(results_list)

    num_data = len(result_data["reward"]["disturb"]["dnn"])

    # result_data["model_error"]["uncertain"]["dnn"][:, 0] = result_data["model_error"]["uncertain"]["dnn"][:, 0] - np.min(result_data["model_error"]["uncertain"]["dnn"][:, 0])
    # result_data["model_error"]["uncertain"]["dnn"][:, 1] = result_data["model_error"]["uncertain"]["dnn"][:, 1] - np.min(result_data["model_error"]["uncertain"]["dnn"][:, 1])
    # result_data["model_error"]["uncertain"]["bnn"][:, 0] = result_data["model_error"]["uncertain"]["bnn"][:, 0] - np.min(result_data["model_error"]["uncertain"]["bnn"][:, 0])
    # result_data["model_error"]["uncertain"]["bnn"][:, 1] = result_data["model_error"]["uncertain"]["bnn"][:, 1] - np.min(result_data["model_error"]["uncertain"]["bnn"][:, 1])

    for i, indicator in enumerate(result_data.keys()):
        for j, case in enumerate(result_data[indicator].keys()):

            if (indicator == "model_error" and case != "noise") or \
                    (indicator != "model_error" and case == "noise"):
                continue

            plt.figure()
            # plt.grid(True)

            if case == "disturb":
                min_case = 0
                max_case = 20
            elif case == "uncertain":
                min_case = -80
                max_case = 80
            else:
                min_case = 0
                max_case = 0.1

            plt.xticks(np.arange(num_data), np.round(np.linspace(min_case, max_case, num_data), 2))
            # plt.xticks(np.round(np.linspace(min_case, max_case, int(num_data/4)+1), 2))

            if indicator == "success_rate":
                get_success_rate_plot(plt, result_data[indicator][case])

            if indicator != "success_rate":
                get_rewards_plot(plt, result_data[indicator][case])

            plt.ylabel(indicator, fontsize=14.0, fontweight='bold')
            plt.title(case, fontsize=14.0, fontweight='bold')
            plt.legend(fontsize=10.0, prop=dict(weight='bold'))

    plt.show()

if __name__ == '__main__':
    main()
