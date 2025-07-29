from random import random
from settings import n
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

for i in range(11, 21):
    victor[i] = 1
    manual[i] = (i + 1) / (2 * i)

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

alpha = victor[n]
manual_error = victor[n] - manual[n]
victor_profiles = [[0] * i + [1 / (n // 2)] * (n - i) for i in range(n + 1)]

# Old operations, kept for reference and compatibility
##################################################
def s(profile):
    return max(sum(profile), 1)

def kick(i, profile):
    return profile[:i] + profile[i + 1 :]  # noqa: E203

def vectorized_kick(profile):
    return [kick(i, profile) for i in range(n)]
##################################################

# New batched operations
##################################################
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


def add_profile(new_p, profiles):
    updated_profiles = []
    for old_p in profiles:
        if sum(abs(x - y) for x, y in zip(old_p, new_p)) >= 0.000001:
            updated_profiles.append(old_p)
    updated_profiles.append(new_p)
    return updated_profiles


def add_two_profiles(new_p1, new_p2, profiles):
    return add_profile(new_p2, add_profile(new_p1, profiles))


def get_random_profiles(count, guided=True):
    # from victor's paper, perhaps this kind of more "guided" profiles are helpful
    def get_random_bid():
        if not guided:
            return random()
        select = random()
        if select <= 1 / 3:
            return 0
        elif select <= 2 / 3:
            return 1 / (n // 2)
        else:
            return random()

    return [list(sorted(get_random_bid() for _ in range(n))) for _ in range(count)]


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
