import os
import neat
import time
import math
import msvcrt
import numpy as np
from itertools import count
from concurrent.futures import ProcessPoolExecutor
import pickle
from functools import partial
import threading

import Pulp_utils
from NEAT_Reproduction import FullyConnectedToOutputReproduction
from settings import (
    n,
    resume,
    epochs_before_analysis,
    time_limit,
    BATCH_SIZE,
    init_alpha_delta,
    mutation_adjustment,
    min_mutation_power,
    max_mutation_power,
    dynamic_mutation,
)
from utility import (
    alpha,
    victor_profiles,
    manual_error,
    s,
    kick,
    s_batch,
    vectorized_kick_batch,
    add_two_profiles,
    get_random_profiles,
)


# Do NOT import matplotlib.pyplot or create GUI/plot objects at import-time.
# Do NOT instantiate NetVisualiser at module import time (it may create GUI objects).
# Both are created inside run_training below so they only exist in the main process.


# get the elapsed time since start_time, format as HH:MM:SS and HH_MM_SS
def format_elapsed_time(start_time):
    """Calculate and format elapsed time as HH:MM:SS or HH_MM_SS format"""
    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    return (
        elapsed_time,
        f"{elapsed_hours:02d}h:{elapsed_minutes:02d}m:{elapsed_seconds:02d}s",
        f"{elapsed_hours:02d}h_{elapsed_minutes:02d}m_{elapsed_seconds:02d}s",
    )


# save / load checkpoint utilities (unchanged)
def save_checkpoint_with_info(population, generation, wcps, filename="checkpoint_full.pkl"):
    checkpoint_data = {"population": population, "generation": generation, "wcps": wcps}
    with open(filename, "wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint_with_info(filename="checkpoint_full.pkl"):
    with open(filename, "rb") as f:
        checkpoint_data = pickle.load(f)
    if not all(k in checkpoint_data for k in ("population", "generation", "wcps")):
        raise ValueError(f"Checkpoint file {filename} is missing required keys.")
    return checkpoint_data["population"], checkpoint_data["generation"], checkpoint_data["wcps"]


# -------------------------
# Evaluation helpers
# -------------------------
def eval_single_genome(
    genome_id, genome, config, worst_case_profiles, alpha, alpha_delta, n, tol=1e-4
):
    """
    Evaluate a single genome. Returns (genome_id, fitness, wca, infeasible_flag, max_violation).
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    infeasible = False
    worst_alpha = alpha
    max_violation = 0.0
    alpha_use = alpha - alpha_delta

    profiles = np.asarray(worst_case_profiles, dtype=float)
    m = profiles.shape[0]
    if m == 0:
        raise ValueError("worst_case_profiles is empty")

    for theta in profiles:
        if len(theta) != n:
            raise ValueError(f"profile length {len(theta)} != expected {n}")

        s_theta = max(np.sum(theta), 1.0)

        # compute S = sum_i net(theta_minus_i)
        S = 0.0
        for i in range(n):
            theta_minus_i = np.concatenate([theta[:i], theta[i + 1 :]])
            h_val = float(net.activate(list(theta_minus_i))[0])
            S += h_val

        left_error = (n - 1) * s_theta - S
        right_error = S - (n - (alpha_use)) * s_theta
        total_error = max(0.0, left_error) + max(0.0, right_error)

        if total_error > max_violation:
            max_violation = total_error

        if total_error > tol:
            infeasible = True
            continue

        achieved_alpha = n - S / s_theta
        if achieved_alpha < worst_alpha:
            worst_alpha = achieved_alpha

    if infeasible:
        fitness = -max_violation
    else:
        fitness = max(0.0, worst_alpha)

    return (genome_id, fitness, worst_alpha, infeasible, max_violation)


# Top-level worker that is picklable by ProcessPoolExecutor
def eval_genome_task(pair, config, worst_case_profiles, alpha, alpha_delta, n):
    """
    pair: (genome_id, genome)
    This wrapper is module-level (not nested) so it can be pickled on spawn.
    """
    gid, genome = pair
    return eval_single_genome(gid, genome, config, worst_case_profiles, alpha, alpha_delta, n)


def eval_genomes_in_parallel(
    genome_pairs, config, worst_case_profiles, alpha, alpha_delta, n, max_workers, chunksize
):
    """
    Evaluates genomes in parallel using ProcessPoolExecutor.
    Writes fitness/wca back into genome objects in parent process after collecting results.
    """
    worst_case_profiles = np.asarray(worst_case_profiles, dtype=float)

    # Partial bind constants; eval_genome_task accepts (pair, config, worst_case_profiles, alpha,...)
    worker = partial(
        eval_genome_task,
        config=config,
        worst_case_profiles=worst_case_profiles,
        alpha=alpha,
        alpha_delta=alpha_delta,
        n=n,
    )

    tasks = [(gid, genome) for (gid, genome) in genome_pairs]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        # map picks each item from `tasks` and passes it as the first (pair) argument to `worker`.
        for res in exe.map(worker, tasks, chunksize=chunksize):
            results.append(res)

    # write results back into genome objects (parent process)
    id_to_genome = {gid: g for (gid, g) in genome_pairs}
    for gid, fitness, worst_alpha, infeasible, max_violation in results:
        genome_obj = id_to_genome[gid]
        genome_obj.fitness = fitness
        genome_obj.wca = worst_alpha

    return


def neat_evaluator(genomes, config, profiles, alpha, alpha_delta, n):
    # Choose number of workers; limit to a reasonable number for your machine
    max_workers = min(os.cpu_count() or 1, 16)
    eval_genomes_in_parallel(
        list(genomes),
        config,
        profiles,
        alpha,
        alpha_delta,
        n,
        max_workers=max_workers,
        chunksize=8,
    )


# Plot update; import matplotlib locally so module import-time won't touch GUI
def update_plot(iterations, ratios, ax, line):
    import matplotlib.pyplot as plt

    line.set_xdata(iterations)
    line.set_ydata(ratios)
    ax.relim()
    ax.autoscale_view()
    max_ratio = max(ratios)
    stats_text = f"Iterations: {len(iterations)}\nMax: {max_ratio:.6f}\nFinal: {ratios[-1]:.6f}"
    [t.remove() for t in ax.texts]
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    plt.pause(0.01)


# Main training loop. Note: NetVisualiser / matplotlib created here in main process only.
def run_training(time_limit_minutes, epochs_per_generation, alpha_delta, population=None, wcp=None):
    # import plotting and visualiser inside the main-run function (so child processes won't import them)
    import matplotlib.pyplot as plt
    from Neat_visual import NetVisualiser

    # Initialize plot for real-time updates (main process only)
    plt.ion()  # interactive mode ON
    fig, ax = plt.subplots(figsize=(12, 8))
    line, = ax.plot([], [], "b-", linewidth=2, label="Current Ratio")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Allocative Efficiency Ratio", fontsize=12)
    ax.set_title("Allocative Efficiency Progress", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    time_limit_seconds = time_limit_minutes * 60

    if population is None:
        pop = neat.Population(neat_config)
    else:
        pop = population

    # Create NetVisualiser in main process only
    try:
        visualiser = NetVisualiser(neat_config)
    except Exception:
        visualiser = None

    best_fitness = -float("inf")
    current_mutation = neat_config.genome_config.weight_mutate_power

    if wcp is None:
        worst_case_profiles = victor_profiles + get_random_profiles(32, n)
    else:
        worst_case_profiles = wcp

    total_error_history = []

    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()

    try:
        max_acheived_alpha = 0.0
        start_time = time.time()

        allocative_ratios = []
        iterations = []

        time_of_last_max_improvement = start_time

        for iter_index in count(start=1):
            iteration_start_time = time.time()

            elapsed_time_check = time.time() - start_time
            if elapsed_time_check >= time_limit_seconds:
                print(f"Time limit of {time_limit_minutes} minutes reached. Stopping training.")
                break

            if keyboard_listener.stop_training:
                print("Training stopped by user input.")
                break

            print("Starting NEAT generation with", len(worst_case_profiles), "profiles")
            print("With", len(pop.species.species), "species and", len(pop.population), "individuals")

            # Run NEAT generation; neat_evaluator will call parallel evaluator from main process.
            winner = pop.run(
                lambda genomes, 
                config: neat_evaluator(genomes, config, worst_case_profiles, alpha, alpha_delta, n),
                1,
            )

            training_duration = time.time() - iteration_start_time

            if iter_index > 1:
                update_plot(iterations, allocative_ratios, ax, line)
                if visualiser is not None:
                    try:
                        visualiser.update(winner)
                    except Exception:
                        pass

            worst_case_start_time = time.time()
            if iter_index % epochs_before_analysis == 1:
                (
                    wcp_left,
                    wcp_right,
                    error_left,
                    error_right,
                    total_error,
                ) = Pulp_utils.worst_case_analysis_neat(genome=winner, config=neat_config, alpha_delta=0.0)

                worst_case_profiles = add_two_profiles(wcp_left, wcp_right, worst_case_profiles)
            else:
                # If not computing new wcp this iteration, use winner's fitness as proxy
                pass #total_error = -min(0.0, winner.fitness) if winner.fitness < 0 else 0.0

            current_alpha = max(0.0, winner.fitness)

            worst_case_duration = time.time() - worst_case_start_time
            iteration_duration = time.time() - iteration_start_time

            should_save_model = total_error < min(total_error_history, default=(100, 0))[0]
            if should_save_model:
                print("💾 New best model found, saving...")
                save_checkpoint_with_info(pop, pop.generation, worst_case_profiles)

            total_error_history.append((total_error, iter_index))

            improvement = winner.fitness - best_fitness
            fitness_updated_mutation = max_mutation_power
            new_alpha_delta = alpha_delta
            
            #updated_mutation = 0.2 * proposed_mutation + 0.8 * current_mutation
            print(f"elapsed_time: {elapsed_time_check:.6g}")
            
            time_updated_mutation = 1 - (elapsed_time_check / time_limit_seconds)
            fitness_updated_mutation = min_mutation_power + (max_mutation_power - min_mutation_power) * (1 - winner.fitness/alpha)
            fitness_updated_mutation = min(fitness_updated_mutation, max_mutation_power)
            print(f"fitness_updated_mutation: {fitness_updated_mutation:.6g}, time_updated_mutation: {time_updated_mutation:.6g}")
            updated_mutation = (fitness_updated_mutation + time_updated_mutation)/2 # take the lower of the two
            updated_mutation = max(min_mutation_power, updated_mutation) # lower bound
            updated_mutation = min(max_mutation_power, updated_mutation) # upper bound
            
            neat_config.genome_config.weight_mutate_power = updated_mutation
            neat_config.genome_config.bias_mutate_power = updated_mutation

            elapsed_time, elapsed_time_str, elapsed_time_underscore = format_elapsed_time(start_time)
            
            print(f"alpha_delta = {alpha_delta:.6g}, mutation = {updated_mutation:.6g}")
            print(f"Current Left-Right error: {error_left:.5f}-{error_right:.5f}")
            print(f"Current total_error: {total_error:.10f}")
            print(f"Winner worst_alpha: {winner.wca:.10f}")
            print(f"winner_fitness: {winner.fitness:.10f}")

            if winner.fitness > 0.0 and (alpha - winner.wca) < 1e-4:
                alpha_delta = alpha_delta / 2

            best_fitness = max(best_fitness, winner.fitness)
            
            if winner.fitness > max_acheived_alpha:
                time_of_last_max_improvement = time.time()
                max_acheived_alpha = winner.fitness

            time_since_improvement = time.time() - time_of_last_max_improvement
            hours_since = int(time_since_improvement // 3600)
            minutes_since = int((time_since_improvement % 3600) // 60)
            seconds_since = int(time_since_improvement % 60)
            time_since_str = f"{hours_since:02d}h:{minutes_since:02d}m:{seconds_since:02d}s"

            allocative_ratios.append(winner.fitness)
            iterations.append(iter_index)

            print(
                "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
                f"\nIteration {iter_index} summary: "
                f"\nelapsed_time={elapsed_time_str} "
                f"\ntime_since_max_improvement={time_since_str} "
                f"\nalpha_delta={alpha_delta:.5f} "
                f"\nbest acheived alpha={max_acheived_alpha:.10f} "
                f"\ntraining_time={training_duration:.2f}s "
                f"\nworst_case_time={worst_case_duration:.2f}s "
                f"\niteration_time={iteration_duration:.2f}s"
                f"\navg_iteration_time={elapsed_time_check/iter_index:.2f}s"
            )

            # target ratio early-exit
            target_ratios = {
                3: 2 / 3,
                4: 2 / 3,
                5: 1 - 1 / (5 * (4 / 120 + 8 / 12)),
                6: 0.868,
                7: 0.748,
                8: 0.755,
                9: 0.772,
                10: 0.882,
            }
            target_ratio = target_ratios.get(n, None)
            if target_ratio is not None and abs(winner.wca - target_ratio) < 0.0001 and alpha_delta == 0.0:
                print(f"Allocative ratio is within tolerance of target {target_ratio:.6f}. Stopping training.")
                break

        return max_acheived_alpha

    finally:
        keyboard_listener.cleanup()


# Keyboard listener
class KeyboardListener:
    def __init__(self):
        self.stop_training = False
        self.listening = False

    def start_listening(self):
        self.listening = True

        def listen():
            while not self.stop_training and self.listening:
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode("utf-8", errors="ignore")
                    if char.lower() == "i":
                        print("\n🛑 Manual stop requested by pressing 'i'")
                        self.stop_training = True
                        break
                time.sleep(0.1)

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def cleanup(self):
        self.listening = False


# -------------------------
# NEAT config
# -------------------------
config_path = "neat_config.ini"
neat_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

neat_config.reproduction_type = FullyConnectedToOutputReproduction


# Entrypoint: ensure safe multiprocessing on Windows
if __name__ == "__main__":
    from multiprocessing import freeze_support, set_start_method

    freeze_support()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    if resume:
        pop, gen, wcp = load_checkpoint_with_info()
        run_training(time_limit * 60, epochs_before_analysis, init_alpha_delta, pop, wcp)
    else:
        run_training(time_limit * 60, epochs_before_analysis, init_alpha_delta)
