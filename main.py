# from old_scripts.train import main as train
# from new_scripts.train_toy import main as train
# from new_scripts.evaluate import different_main as evaluate
# from new_scripts.mnist import main as train
from new_scripts.plot_graph import different_main as plot
# from new_scripts.retrain_toy import main as retrain
# from tqdm import tqdm
# import os

# from old_scripts.generate import main as generate
# from old_scripts.eval import main as evaluate

# paths = "chkpts/mnist"
# for path in tqdm(os.listdir(paths)):
#     if path.endswith(".pt"):
#         print(path)
#         generate(os.path.join(paths, path))
#         evaluate(os.path.join(paths, path))

# retrain()
# train()
# evaluate()
plot()