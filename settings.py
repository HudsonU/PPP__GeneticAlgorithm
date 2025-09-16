# everything in this file is used to change GA/project settings
import os
import numpy as np
from random import seed
from datetime import datetime, timezone

# number of agents
n = 3

# set time limit for each episode in hours
time_limit = 12

# number of worst case profiles to keep in memory
BATCH_SIZE = 64

# intial slack on alpha for mip and fitness
init_alpha_delta = 0.0

# seed for the environment, used to ensure reproducibility
env_seed = 1121111

# continue training from a previous run
resume = False

# number of generations to train before saving a model
# and adding the worst case profile to the list
epochs_before_analysis = 1

# network configuration, refers to hidden layers and nodes
min_layers = 1
max_layers = 3
min_nodes_per_layer = 2
max_nodes_per_layer = 50

########################## might try later #################################
# relative costs for the network
# e.g. if 2 networks have the same worst case, the one with more nodes will be penalized
# e.g. final score = worst_case - cost_per_node * number_of_nodes
# cost_per_node = 0.0001
#############################################################################

# Setting the date and time 
dt = datetime.now(timezone.utc)

# setting the environment seeds
seed(env_seed) # python + NEAT random seed
np.random.seed(env_seed) # numpy random seed

###################### WHAT DOES STORY DO??? ######################
# shapes can be ignored for neat
try:
    story = int(os.environ["story"])
except KeyError:
    print("by default, set story to 0, i.e., the playground")
    story = 0

try:
    env_shapes_raw = os.environ["shapes"]
    env_shapes = [n - 1] + [int(x) for x in env_shapes_raw.split("-")] + [1]
except KeyError:
    env_shapes_raw = None
    env_shapes = None

try:
    env_size = int(os.environ["size"])
except KeyError:
    env_size = 2


def get_timestamp():
    format_str = "%b_%d_%Hhr_%Mmin_%Ssec"
    #assert False, "anonymized timezone"
    result = datetime.now(timezone.utc).strftime(format_str)
    return result


finger_print = f"{get_timestamp()}-{n}-{story}-{env_seed}-{env_shapes_raw}"
print(finger_print)

# WHAT DO THESE MEAN???
# if story == 3:
#     env_alpha_delta = int(os.environ["alphadelta"])
#     env_train_using_worst_case = int(os.environ["useworstcase"])
#     env_train_using_free_samples = int(os.environ["usefreesamples"])
#     story3_worst_case_done_limit = None
# else:
#     env_alpha_delta = None
#     env_train_using_worst_case = None
#     env_train_using_free_samples = None
#     story3_worst_case_done_limit = None

# if story == 9:
#     story9_reinitialize_whole_network = int(os.environ["reinit"])
#     story9_small = int(os.environ["s9small"])
#     story9_prune_node = int(os.environ["prunenode"])
#     story9_lower_alpha_delta = int(os.environ["loweralphadelta"])
# else:
#     story9_reinitialize_whole_network = None
#     story9_small = None
#     story9_prune_node = None
#     story9_lower_alpha_delta = None
