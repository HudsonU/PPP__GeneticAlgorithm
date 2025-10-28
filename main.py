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
from Neat_evaluation import neat_evaluator
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
    saves_per_run
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
    save_array_to_csv,
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

# Plot update; import matplotlib locally so module import-time won't touch GUI
def update_plot(iterations, series_list, ax, lines, labels=None, save_path: str | None = None, dpi: int = 150, transparent: bool = False):
    """
    iterations : list[int]
    series_list: list of lists, e.g. [winner_fitness_history, winner_wca_history, winner_vio_history]
    ax         : matplotlib Axes
    lines      : list of Line2D objects corresponding to series_list
    labels     : optional list of labels for stats text order
    """
    import matplotlib.pyplot as plt

    if len(iterations) == 0 or not any(len(s) for s in series_list):
        return

    # ensure lines and series align
    if len(series_list) != len(lines):
        raise AssertionError("series_list and lines must be same length")

    # update each line
    for series, line in zip(series_list, lines):
        line.set_xdata(iterations)
        line.set_ydata(series)

    ax.relim()
    ax.autoscale_view()

    # build stats text for each series
    stats_lines = []
    if labels is None:
        labels = [f"Series {i}" for i in range(len(series_list))]

    for label, series in zip(labels, series_list):
        if len(series) > 0:
            max_v = max(series)
            final_v = series[-1]
            stats_lines.append(f"{label}  Max: {max_v:.6f}  Final: {final_v:.6f}")
        else:
            stats_lines.append(f"{label}  (no data)")

    stats_text = f"Iterations: {len(iterations)}\n" + "\n".join(stats_lines)

    # remove previous text objects safely
    for t in list(ax.texts):
        try:
            t.remove()
        except Exception:
            pass

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # draw to update the GUI
    plt.pause(0.01)

    # save if requested
    if save_path:
        # ensure extension
        if not save_path.lower().endswith(".png"):
            save_path = save_path + ".png"
        try:
            fig = ax.figure
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
            print(f"Saved plot to: {save_path}")
        except Exception as e:
            print(f"Warning: failed to save plot to {save_path}: {e}")


# Neat evaluator wrapper to use optimized evaluator

# Main training loop. Note: NetVisualiser / matplotlib created here in main process only.
def run_training(time_limit_minutes, epochs_per_generation, alpha_delta, population=None, wcp=None):
    # import plotting and visualiser inside the main-run function (so child processes won't import them)
    import matplotlib.pyplot as plt
    from Neat_visual import NetVisualiser

    # Initialize plot for real-time updates (main process only)
    plt.ion()  # interactive mode ON
    fig, ax = plt.subplots(figsize=(12, 8))
    line, = ax.plot([], [], "b-", linewidth=2, label="Current Ratio")
    line_wca, = ax.plot([], [], "g--", linewidth=1.5, label="Winner WCA")
    line_vio, = ax.plot([], [], "r:", linewidth=1.5, label="Winner VIO")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Allocative Efficiency Ratio / WCA / VIO", fontsize=12)
    ax.set_title("Allocative Efficiency Progress", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    lines = [line, line_wca, line_vio]

    time_limit_seconds = time_limit_minutes * 60

    if population is None:
        pop = neat.Population(neat_config)
    else:
        pop = population

    max_workers = min(os.cpu_count() or 1, 16)
    evaluator = neat_evaluator(n, neat_config, max_workers=max_workers)
    
    # Create NetVisualiser in main process only
    visualiser = NetVisualiser(neat_config)

    best_fitness = -float("inf")
    current_mutation = neat_config.genome_config.weight_mutate_power

    if wcp is None:
        worst_case_profiles = victor_profiles + get_random_profiles(256, n)
    else:
        worst_case_profiles = wcp

    total_error_history = []

    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()

    try:
        max_acheived_fitness = -100
        start_time = time.time()

        winner_fitness_history = []
        winner_wca_history = []
        winner_vio_history = []
        iterations = []

        time_of_last_max_improvement = start_time
        
        # set up saving intervals
        interval = time_limit_seconds / max(1, saves_per_run)
        next_save_time = interval
        saves_done = 0
        min_gens_between_saves = 1
        last_save_iter = -min_gens_between_saves


        for iter_index in count(start=1):
            iteration_start_time = time.time()

            elapsed_time_check = time.time() - start_time
            if elapsed_time_check >= time_limit_seconds or keyboard_listener.stop_training:
                update_plot(
                    iterations,
                    [winner_fitness_history, winner_wca_history, winner_vio_history],
                    ax,
                    lines,
                    labels=["Fitness", "WCA", "VIO"],
                    save_path=f"saves/p_{n}a_fit{winner.fitness:.6f}.png",
                )
                try:
                    visualiser.save_png(f"saves/n_{n}a_iter{iter_index}_t{int(elapsed_time_check)}s_fit{winner.fitness:.6f}.png", dpi=150, transparent=True)
                except Exception:
                    pass
                save_array_to_csv(
                    np.array(list(zip(winner_fitness_history, winner_wca_history, winner_vio_history))),
                    f"saves/f_{n}a_fit{winner.fitness:.6f}.csv",
                )
                save_checkpoint_with_info(pop, pop.generation, worst_case_profiles)
                break

            print("Starting NEAT generation with", len(worst_case_profiles), "profiles")
            print("With", len(pop.species.species), "species and", len(pop.population), "individuals")

            # Run NEAT generation; neat_evaluator will call parallel evaluator from main process.
            winner = evaluator.run(
                pop,
                worst_case_profiles,
                alpha,
                alpha_delta,
                epochs_per_generation,
            )

            training_duration = time.time() - iteration_start_time

            if iter_index > 1:
                update_plot(
                    iterations,
                    [winner_fitness_history, winner_wca_history, winner_vio_history],
                    ax,
                    lines,
                    labels=["Fitness", "WCA", "VIO"],
                )
                if visualiser is not None:
                    try:
                        visualiser.update(winner)
                    except Exception:
                        pass

            worst_case_start_time = time.time()
            if iter_index % epochs_before_analysis == 1 or epochs_before_analysis == 1:
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

            total_error_history.append((total_error, iter_index))

            #improvement = winner.fitness - best_fitness
            new_alpha_delta = alpha_delta
            
            #updated_mutation = 0.2 * proposed_mutation + 0.8 * current_mutation
            #print(f"elapsed_time: {elapsed_time_check:.6g}")
            
            time_updated_mutation = min_mutation_power + (1 - elapsed_time_check / time_limit_seconds) * (max_mutation_power - min_mutation_power)
            fitness_updated_mutation = min_mutation_power + (max_mutation_power - min_mutation_power) * abs(winner.fitness)
            
            print(f"fitness_updated_mutation: {fitness_updated_mutation:.6g}, time_updated_mutation: {time_updated_mutation:.6g}")
            updated_mutation = (fitness_updated_mutation + time_updated_mutation)/2 # take the average of the two
            updated_mutation = max(min_mutation_power, updated_mutation) # lower bound
            updated_mutation = min(max_mutation_power, updated_mutation) # upper bound
            
            neat_config.genome_config.weight_mutate_power = updated_mutation
            neat_config.genome_config.bias_mutate_power = updated_mutation

            elapsed_time, elapsed_time_str, elapsed_time_underscore = format_elapsed_time(start_time)
            
            print(f"alpha_delta = {alpha_delta:.6g}, mutation = {updated_mutation:.6g}")
            print(f"Current Left-Right error: {error_left:.5f} ~~~ {error_right:.5f}")
            print(f"Current total_error: {total_error:.10f}")
            print(f"Winner worst_alpha: {winner.wca:.10f}")
            print(f"Winner vio: {winner.vio:.10f}")
            print(f"winner_fitness: {winner.fitness:.10f}")

            if visualiser is not None and saves_done < saves_per_run:
                # elapsed_time_check is already computed as time.time() - start_time
                if elapsed_time_check >= next_save_time:
                    save_checkpoint_with_info(pop, pop.generation, worst_case_profiles)
                    try:
                        visualiser.save_png(f"saves/n_{n}a_iter{iter_index}_t{int(elapsed_time_check)}s_fit{winner.fitness:.6f}.png", dpi=150, transparent=True)
                    except Exception:
                        pass
                    saves_done += 1
                    last_save_iter = iter_index
                    next_save_time += interval


            if winner.fitness > 0.0:
                alpha_delta = alpha_delta * 0.95

            best_fitness = max(best_fitness, winner.fitness)
            
            if winner.fitness > max_acheived_fitness:
                time_of_last_max_improvement = time.time()
                max_acheived_fitness = winner.fitness

            time_since_improvement = time.time() - time_of_last_max_improvement
            hours_since = int(time_since_improvement // 3600)
            minutes_since = int((time_since_improvement % 3600) // 60)
            seconds_since = int(time_since_improvement % 60)
            time_since_str = f"{hours_since:02d}h:{minutes_since:02d}m:{seconds_since:02d}s"

            winner_fitness_history.append(winner.fitness)
            winner_wca_history.append(winner.wca)
            winner_vio_history.append(winner.vio)
            iterations.append(iter_index)

            print(
                "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
                f"\nIteration {iter_index} summary: "
                f"\nelapsed_time={elapsed_time_str} "
                f"\ntime_since_max_improvement={time_since_str} "
                f"\nalpha_delta={alpha_delta:.5f} "
                f"\nbest acheived alpha={max_acheived_fitness:.10f} "
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
            if target_ratio is not None and abs(winner.fitness - target_ratio) < 0.0001 and alpha_delta == 0.0:
                print(f"Allocative ratio is within tolerance of target {target_ratio:.6f}. Stopping training.")
                update_plot(
                    iterations,
                    [winner_fitness_history, winner_wca_history, winner_vio_history],
                    ax,
                    lines,
                    labels=["Fitness", "WCA", "VIO"],
                    save_path=f"saves/p_{n}a_fit{winner.fitness:.6f}.png",
                )
                try:
                    visualiser.save_png(f"saves/n_{n}a_iter{iter_index}_t{int(elapsed_time_check)}s_fit{winner.fitness:.6f}.png", dpi=150, transparent=True)
                except Exception:
                    pass
                save_array_to_csv(
                    np.array(list(zip(winner_fitness_history, winner_wca_history, winner_vio_history))),
                    f"saves/f_{n}a_fit{winner.fitness:.6f}.csv",
                )
                save_checkpoint_with_info(pop, pop.generation, worst_case_profiles)
                break

        return max_acheived_fitness

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
