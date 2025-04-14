import os
import matplotlib.pyplot as plt
from collections import defaultdict



def main():
    # dataset: gaussian8, l: 0.25, iteration: 1, wasserstein distance: 0.31704983997444014, precision: [0.995], recall: [0.99]
    path = "test1.txt"
    path_to_save_wasser = "plots/wasser/"
    path_to_save_precision = "plots/precision/"
    path_to_save_recall = "plots/recall/"

    if not os.path.exists(path_to_save_wasser): os.makedirs(path_to_save_wasser)
    if not os.path.exists(path_to_save_precision): os.makedirs(path_to_save_precision)
    if not os.path.exists(path_to_save_recall): os.makedirs(path_to_save_recall)

    with open(path, "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split(", ") for x in data]
    grouped_data = defaultdict(list)

    for line in data:
        grouped_data[", ".join(line[:2])].append(line)

    iterations = lambda data: [int(x[2].split(": ")[1]) for x in data]
    wasserstein_distances = lambda data: [float(x[3].split(": ")[1]) for x in data]
    precision = lambda data: [float(x[4].split(": ")[1][1:-1]) for x in data]
    recall = lambda data: [float(x[5].split(": ")[1][1:-1]) for x in data]
    
    for key, value in grouped_data.items():
        key = key.replace(", ", "_")
        key = key.replace(": ", "_")
        plt.figure(key)
        plt.scatter(iterations(value), wasserstein_distances(value), label=key)
        plt.xlabel("Iterations")
        plt.ylabel("Wasserstein Distance")
        plt.title(f"Wasserstein Distance vs Iterations")
        plt.savefig(rf"{path_to_save_wasser}//{key}.png")
        plt.close()

    for key, value in grouped_data.items():
        key = key.replace(", ", "_")
        key = key.replace(": ", "_")
        plt.figure(key)
        plt.scatter(iterations(value), precision(value), label=key)
        plt.xlabel("Iterations")
        plt.ylabel("Precision")
        plt.title(f"Precision vs Iterations")
        plt.savefig(rf"{path_to_save_precision}//{key}.png")
        plt.close()

    for key, value in grouped_data.items():
        key = key.replace(", ", "_")
        key = key.replace(": ", "_")
        plt.figure(key)
        plt.scatter(iterations(value), recall(value), label=key)
        plt.xlabel("Iterations")
        plt.ylabel("Recall")
        plt.title(f"Recall vs Iterations")
        plt.savefig(rf"{path_to_save_recall}//{key}.png")
        plt.close()
        


def different_main():

    path = "test4.txt"

    with open(path, "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split(", ") for x in data]
    precision = [None] * 6
    wasserstein = [None] * 6
    recall = [None] * 6
    for line in data:
        iteration = int(line[2].split(": ")[1])
        precision[iteration] = float(line[4].split(": ")[1][1:-1])
        wasserstein[iteration] = float(line[3].split(": ")[1])
        recall[iteration] = float(line[5].split(": ")[1][1:-1])


    plt.figure("Precision")
    plt.plot(precision)
    plt.xlabel("Iterations")
    plt.ylabel("Precision")
    plt.title("Precision vs Iterations")
    plt.savefig("plots/precision.png")

    plt.figure("Wasserstein")
    plt.plot(wasserstein)
    plt.xlabel("Iterations")
    plt.ylabel("Wasserstein Distance")
    plt.title("Wasserstein Distance vs Iterations")
    plt.savefig("plots/wasserstein.png")

    plt.figure("Recall")
    plt.plot(recall)
    plt.xlabel("Iterations")
    plt.ylabel("Recall")
    plt.title("Recall vs Iterations")
    plt.savefig("plots/recall.png")




    


if __name__ == '__main__':
    main()