from random import random
from settings import n as n_agents
import numpy as np

# Victor's conjectured upperbound: only kept 3 significant digits
victor = {
    3: 2 / 3,
    4: 2 / 3,
    # 5: 0.714,
    5: 1 - 1 / (5 * (4 / 120 + 8 / 12)),
    6: 0.868,
    7: 0.748,
    8: 0.755,
    9: 0.772,
    10: 0.882,
}

# best manual values for n agents, only kept 3 significant digits
manual = {
    3: 2 / 3,
    4: 0.625,
    5: 0.600,
    6: 0.583,
    7: 0.571,
    8: 0.563,
    9: 0.556,
    10: 0.550,
}

# best manual values for n agents based on guo's paper
for i in range(11, 21):
    victor[i] = 1
    manual[i] = (i + 1) / (2 * i)

#???
achieved_by_story9 = {
    3: 0,
    4: 0,
    5: 0,
    6: 0.16622897275865772,
    7: 0.04890144569435921,
    8: 0.06989498909658096,
    9: 0.09840350518150806,
    10: 0.20024883504189483,
}
#???
achieved_by_story3_1hr = {
    3: 0,
    4: 0,
    5: 0,
    6: 0.19060998774269344,
    7: 0.08880394658095211,
    8: 0.09733750076467262,
    9: 0.13288543144923703,
    10: 0.25759407474829893,
}

# alpha = the conjectured upperbound
alpha = victor[n_agents]

# manual = the best known manual solution
manual_error = victor[n_agents] - manual[n_agents]

# initial profiles: from all 0 to all 1/(n/2), step 1/(n/2)
victor_profiles = [[0] * i + [1 / ((n_agents) // 2)] * ((n_agents) - i) for i in range(n_agents)]

# Old operations, kept for reference and compatibility
##################################################
def s(profile):
    return max(sum(profile), 1)

def kick(i, profile):
    return profile[:i] + profile[i + 1 :]  # noqa: E203

def vectorized_kick(profile):
    return [kick(i, profile) for i in range(n_agents)]
##################################################

# New batched operations
##################################################

# takes in a batch of profiles, returns a column vector of s values
def s_batch(profiles):
    profiles = np.asarray(profiles)
    s_values = np.maximum(np.sum(profiles, axis=1), 1)
    return s_values.reshape(-1, 1)

def vectorized_kick_batch(profiles):
    profiles = np.asarray(profiles)
    n = profiles.shape[1]
    idx = np.arange(n)
    kicked = np.stack([np.delete(profiles, i, axis=1) for i in idx], axis=1)
    return kicked
##################################################

# called by add_two_profiles, chechs if new_p is sufficiently different from existing profiles
def add_profile(new_p, profiles, minDif=1e-6):
    new_p = np.array(new_p, dtype=float)
    if len(profiles) == 0:
        return [new_p]

    profiles_np = np.array(profiles, dtype=float)

    # Compute distance (sum of absolute differences) for each old profile
    diffs = np.sum(np.abs(profiles_np - new_p), axis=1)

    # Keep only profiles that are different enough
    mask = diffs >= minDif
    updated_profiles = profiles_np[mask].tolist()

    # Add the new profile
    updated_profiles.append(new_p.tolist())
    return updated_profiles

# adds two profiles to the list
def add_two_profiles(new_p1, new_p2, profiles):
    return add_profile(new_p2, add_profile(new_p1, profiles))

# init the random profiles
def get_random_profiles(count, guided=True):
    # from victor's paper, perhaps this kind of more "guided" profiles are helpful
    def get_random_bid():
        if not guided:
            return random()
        select = random()
        if select <= 1 / 3:
            return 0
        elif select <= 2 / 3:
            return 1 / ((n_agents) // 2)
        else:
            return random()

    return [list(sorted(get_random_bid() for _ in range(n_agents))) for _ in range(count)]

# NOT USED, NOT SURE WHAT IT DID
def loss_schedule(train_count):
    res = 0
    for i in range(1, train_count + 1):
        if i <= 10:
            res += 0.0001
        elif i <= 20:
            res += 0.0002
        elif i <= 30:
            res += 0.0004
        elif i <= 40:
            res += 0.0008
        elif i <= 50:
            res += 0.0016
        elif i <= 60:
            res += 0.0032
        elif i <= 70:
            res += 0.0064
        elif i <= 80:
            res += 0.0128
        elif i <= 90:
            res += 0.0256
        elif i <= 100:
            res += 0.0512
        else:
            res += 0.1024
    return res
