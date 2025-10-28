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
BATCH_SIZE = 10000

# dynamic mutation settings
dynamic_mutation = True
min_mutation_power = 0.01
max_mutation_power = 1.0
mutation_adjustment = 0.1  # NOT USED: how much to change mutation rate by (cur*(1 +/- mutation_adjustment))

# intial slack on alpha for mip and fitness
init_alpha_delta = 0.0

# seed for the environment, used to ensure reproducibility
env_seed = 322343321

# continue training from a previous run
resume = False

# number of generations to train before saving a model
# and adding the worst case profile to the list
epochs_before_analysis = 1

# how many saves to make per run (save every time_limit/spr hours)
saves_per_run = 100

# NOT USED network configuration, refers to hidden layers and nodes
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

def get_timestamp():
    format_str = "%b_%d_%Hhr_%Mmin_%Ssec"
    #assert False, "anonymized timezone"
    result = datetime.now(timezone.utc).strftime(format_str)
    return result


#finger_print = f"{get_timestamp()}-{n}-{story}-{env_seed}-{env_shapes_raw}"
#print(finger_print)

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
