import torch
import os
import datetime
import numpy as np
from zoneinfo import ZoneInfo
from random import seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

try:
    env_seed = int(os.environ["seed"])
except KeyError:
    env_seed = 0

seed(env_seed)
torch.manual_seed(env_seed)
np.random.seed(env_seed)

n = int(os.environ["n"])

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
    result = datetime.datetime.now(ZoneInfo("UTC")).strftime(format_str)
    return result


finger_print = f"{get_timestamp()}-{n}-{story}-{env_seed}-{env_shapes_raw}"
print(finger_print)

if story == 3:
    env_alpha_delta = int(os.environ["alphadelta"])
    env_train_using_worst_case = int(os.environ["useworstcase"])
    env_train_using_free_samples = int(os.environ["usefreesamples"])
    story3_worst_case_done_limit = None
else:
    env_alpha_delta = None
    env_train_using_worst_case = None
    env_train_using_free_samples = None
    story3_worst_case_done_limit = None

if story == 9:
    story9_reinitialize_whole_network = int(os.environ["reinit"])
    story9_small = int(os.environ["s9small"])
    story9_prune_node = int(os.environ["prunenode"])
    story9_lower_alpha_delta = int(os.environ["loweralphadelta"])
else:
    story9_reinitialize_whole_network = None
    story9_small = None
    story9_prune_node = None
    story9_lower_alpha_delta = None

try:
    resume_filename = os.environ["resume"]
except KeyError:
    resume_filename = None
