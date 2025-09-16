import os
import neat
import time
import matplotlib.pyplot as plt
import math
import msvcrt
import time
import numpy as np
from itertools import count
import pickle
import Pulp_utils
import threading
from NEAT_Reproduction import FullyConnectedToOutputReproduction
from settings import (
    n,
    finger_print,
    resume,
    epochs_before_analysis,
    time_limit,
    BATCH_SIZE,
    init_alpha_delta
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
# Initialize plot for real-time updates
plt.ion()  # interactive mode ON
fig, ax = plt.subplots(figsize=(12, 8))
line, = ax.plot([], [], "b-", linewidth=2, label="Current Ratio")
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Allocative Efficiency Ratio", fontsize=12)
ax.set_title("Allocative Efficiency Progress", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend() 

# Load NEAT configuration
config_path = "neat_config.ini"
neat_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)
#neat_config.reproduction_type = FullyConnectedToOutputReproduction
# get the elapsed time since start_time, format as HH:MM:SS and HH_MM_SS
def format_elapsed_time(start_time):
    """Calculate and format elapsed time as HH:MM:SS or HH_MM_SS format"""
    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    return elapsed_time, f"{elapsed_hours:02d}h:{elapsed_minutes:02d}m:{elapsed_seconds:02d}s", f"{elapsed_hours:02d}h_{elapsed_minutes:02d}m_{elapsed_seconds:02d}s"

# save the population with extra info
def save_checkpoint_with_info(population, generation, wcps, filename='checkpoint_full.pkl'):
    """
    Save NEAT population + extra info in a single file.
    
    population : neat.Population
    generation : int
    wcps : dict 
    filename   : str
    """
    checkpoint_data = {
        'population': population,
        'generation': generation,
        'wcps': wcps
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint_with_info(filename='checkpoint_full.pkl'):
    """
    Load NEAT population + extra info from a checkpoint file.

    Returns
    -------
    population : neat.Population
        The saved NEAT population.
    generation : int
        The saved generation number.
    wcps : dict
        The saved worst-case profiles.
    """
    with open(filename, 'rb') as f:
        checkpoint_data = pickle.load(f)

    # Defensive checks
    if not all(k in checkpoint_data for k in ('population', 'generation', 'wcps')):
        raise ValueError(f"Checkpoint file {filename} is missing required keys.")

    return checkpoint_data['population'], checkpoint_data['generation'], checkpoint_data['wcps']


def eval_genomes(genomes, config, worst_case_profiles, alpha, alpha_delta, n):
    """
    Evaluate GA candidates for VCG redistribution efficiency.
    """
    if not worst_case_profiles:
        raise ValueError("worst_case_profiles is empty")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        infeasible = False
        worst_ratio = float("inf")
        max_violation = 0.0

        for theta in worst_case_profiles:
            theta = np.asarray(theta, dtype=float)
            if len(theta) != n:
                raise ValueError(f"Profile length {len(theta)} != expected {n}")

            # First-best welfare for this profile
            s_theta = max(np.sum(theta), 1.0)

            # Total redistributed payments S(theta)
            S = 0.0
            for i in range(n):
                theta_minus_i = np.concatenate([theta[:i], theta[i+1:]])
                h_val = float(net.activate(list(theta_minus_i))[0])
                S += h_val

            # Check inequality bounds for feasibility
            left_error = (n - 1) * s_theta - S         # should be ≤ 0
            right_error = S - (n - alpha) * s_theta    # should be ≤ 0
            total_error = max(0.0, left_error) + max(0.0, right_error)

            max_violation = max(max_violation, total_error)

            if total_error > 1e-3 + alpha_delta:
                infeasible = True
                continue

            # Compute allocative efficiency ratio
            achieved = n * s_theta - S
            efficiency_ratio = achieved / (s_theta + 1e-9)

            # Track worst-case (minimum) ratio
            if efficiency_ratio < worst_ratio:
                worst_ratio = efficiency_ratio

        # Assign genome fitness
        if infeasible:
            # Penalize infeasible networks by negative of max violation
            genome.fitness = -max_violation
        else:
            genome.fitness = worst_ratio

        # Store worst-case ratio for logging/debugging
        genome.wcr = worst_ratio
    

# sets up time limit, sets up network and training vars
# begins main training loop, 
# CHANGE THIS -> each loops runs hypernetwork_train, then worst-case analysis
# updates profiles, saves models and plots results
def run_training(time_limit_minutes, epochs_per_generation, alpha_delta, population=None,wcp=None):
    time_limit_seconds = time_limit_minutes * 60
    #############################################################
    #############################################################
    #############################################################
    # Initialize the neat population
    if population == None:
        pop = neat.Population(neat_config)
    else:
        pop = population
    #pop.add_reporter(neat.StdOutReporter(True)) 
    #stats = neat.StatisticsReporter()
    #pop.add_reporter(stats)

    # initialise the profiles
    #############################################################
    #############################################################
    #############################################################
    
    # add random profiles
    if wcp == None:
        worst_case_profiles = victor_profiles
    else:
        worst_case_profiles = wcp
    #worst_case_profiles = get_random_profiles(BATCH_SIZE, guided=True)
    total_error_history = []
    
    # Get alpha delta
    achievable_alpha_delta = alpha_delta
    
    #alpha_delta = max(achievable_alpha_delta - 0.001, 0)
    #############################
    print(
        f"fingerprint {finger_print} alpha_delta={alpha_delta}"
    )
    
    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()
    
    ######################
    # Main training loop #
    ######################
    try:
        max_allocative_ratio = -999999999999
        start_time = time.time()
        
        allocative_ratios = []
        iterations = []
        
        # used to check if training hasn't improved max ratio in last 2 hours
        time_of_last_max_improvement = start_time
        
        # infinite loop with iter_index starting at 1, breaks on time limit or manual stop
        for iter_index in count(start=1):
            iteration_start_time = time.time()
            
            # Check stopping conditions
            elapsed_time_check = time.time() - start_time
            if elapsed_time_check >= time_limit_seconds:
                print(
                    f"Time limit of {time_limit_minutes} minutes reached. Stopping training."
                )
                break
            #############################
            
            # Check for manual stop
            if keyboard_listener.stop_training:
                print("Training stopped by user input.")
                break
            
            # Train the hypernetwork
            training_start_time = time.time()
            # avg_loss = hypernetwork_train(
            #     hypernetwork,
            #     target_network,
            #     optimizer,
            #     profiles,
            #     alpha_delta=alpha_delta,
            #     device=device,
            #     epochs=500,
            # )
            
            print("Starting",epochs_per_generation,"NEAT generations with",len(worst_case_profiles),"profiles")
            print("With", len(pop.species.species), "species and",len(pop.population),"individuals")
            # if iter_index > 1:
            #     while abs(winner.fitness) < alpha_delta:
            #         winner = pop.run(lambda genomes, config: eval_genomes(genomes, config, worst_case_profiles, alpha_delta), epochs_per_generation)
            # else:
            winner = pop.run(lambda genomes, config: eval_genomes(genomes, config, worst_case_profiles[-BATCH_SIZE:],alpha,alpha_delta,n), epochs_per_generation)
            #print("NEAT generations complete.")
            #print(winner)
            training_duration = time.time() - training_start_time
            #############################
            
            # Running worst case analysis
            worst_case_start_time = time.time()
            if True:#winner.fitness > 0:
                (
                    wcp_left,
                    wcp_right,
                    error_left,
                    error_right,
                    total_error,
                ) = Pulp_utils.worst_case_analysis_neat(genome=winner,config=neat_config,alpha_delta=alpha_delta)
                
                worst_case_profiles = add_two_profiles(wcp_left, wcp_right, worst_case_profiles)
            
            #total_error = winner.fitness
            current_ratio = winner.wcr
                
            worst_case_duration = time.time() - worst_case_start_time
            #############################

            iteration_duration = time.time() - iteration_start_time   
            ############################# 
            
            # Save model 
            should_save_model = total_error < min(total_error_history, default=(100, 0))[0]
            if should_save_model:
                print("💾 New best model found, saving...")
                save_checkpoint_with_info(pop,pop.generation,worst_case_profiles)

            total_error_history.append((total_error, iter_index))
            #achievable_alpha_delta = min(achievable_alpha_delta, total_error)

            # Adjust alpha_delta based on performance
            if current_ratio - alpha_delta < 0.001:
                new_alpha_delta = max(alpha_delta / 2, alpha - 0.01)
            else:
                new_alpha_delta = (alpha_delta + achievable_alpha_delta) / 2
            new_alpha_delta = max(min(new_alpha_delta, achievable_alpha_delta - 0.001), 0)
            print(f"alpha_delta goes from {alpha_delta} to {new_alpha_delta}")
            alpha_delta = new_alpha_delta
            #############################
            
            # Calculate elapsed time from the very start
            elapsed_time, elapsed_time_str, elapsed_time_underscore = format_elapsed_time(start_time)
            
            # Check current allocative ratio (total_error) vs max allocative ratio (alpha)
            # total_error should be the worst-case total_error (smaller is better)
            #current_ratio = max(0.0, total_error)  # allocative ratio, higher better
            print(f"Current total_error: {total_error:.10f}")
            print(f"allocative_ratio: {current_ratio:.10f}")
            print(f"winner_fitness: {winner.fitness:.10f}")

            # MAY NOT BE NEEDED FOR NEAT - Save model if improved
            # if should_save_model:
            #     os.makedirs("hypernets", exist_ok=True)
            #     timestamp = time.strftime("%b_%d_%Hhr_%Mmin_%Ssec")
            #     hn_filename = f"hypernets/{timestamp}-n{n}-ratio{current_ratio:.8f}-time{elapsed_time_str}-{finger_print}.hn"
                
            #     save_data = {
            #         'hypernetwork_state_dict': hypernetwork.state_dict(),
            #         'max_allocative_ratio': current_ratio,
            #         'elapsed_time_seconds': elapsed_time,
            #         'elapsed_time_formatted': elapsed_time_str,
            #         'n': n,
            #         'finger_print': finger_print,
            #         'target_arch': target_arch,
            #         'hn_config': hn_config,
            #         'timestamp': timestamp,
            #         'iteration': iter_index,
            #         'total_error': total_error
            #     }
                
            #     torch.save(save_data, hn_filename)
            
            # Update max allocative ratio and time since last improvement
            if winner.fitness > max_allocative_ratio:
                time_of_last_max_improvement = time.time()
                max_allocative_ratio = winner.fitness
                
            # Calculate time since last max improvement
            time_since_improvement = time.time() - time_of_last_max_improvement
            hours_since = int(time_since_improvement // 3600)
            minutes_since = int((time_since_improvement % 3600) // 60)
            seconds_since = int(time_since_improvement % 60)
            time_since_str = f"{hours_since:02d}h:{minutes_since:02d}m:{seconds_since:02d}s"
            #############################
            
            # Check if no improvement for more than 2 hours
            # if time_since_improvement > 20 * 1800:  # 2 hours in seconds
            #     print(f"⏰ No improvement in max allocative ratio for {time_since_str}. Stopping training.")
            #     break
            #############################
            # Record allocative ratio for plotting
            allocative_ratios.append(winner.fitness)
            iterations.append(iter_index)
            
            # Print iteration summary
            print(
                "=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+="
                f"\nIteration {iter_index:03d} summary: "
                f"\nelapsed_time={elapsed_time_str} "
                f"\ntime_since_max_improvement={time_since_str} "
                # f"error_left={error_left:.20f} "
                # f"error_right={error_right:.20f} "
                f"\nalpha_delta={alpha_delta:.20f} "
                #f"final_loss={avg_loss:.20f} "
                f"\nmax_allocative_ratio={max_allocative_ratio:.20f} "
                f"\ntraining_time={training_duration:.2f}s "
                f"\nworst_case_time={worst_case_duration:.2f}s "
                f"\niteration_time={iteration_duration:.2f}s"
            )
            print(f"\nCurrent worst case allocative ratio: {current_ratio}")
            update_plot(iterations, allocative_ratios, ax, line)
            
            # Break if allocative ratio is close to target value
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
            if abs(current_ratio - target_ratio) < 0.0001:
                print(f"Allocative ratio {current_ratio:.6f} is within 0.0001 of target {target_ratio:.6f}. Stopping training.")
                #break
        
       
    
        return max_allocative_ratio
    
    finally:
        keyboard_listener.cleanup()

# Plot allocative efficiency ratios over iterations
def update_plot(iterations, ratios, ax, line):
    # Update line data
    line.set_xdata(iterations)
    line.set_ydata(ratios)

    # Rescale axes
    ax.relim()
    ax.autoscale_view()

    # Update stats box
    max_ratio = max(ratios)
    stats_text = f"Iterations: {len(iterations)}\nMax: {max_ratio:.6f}\nFinal: {ratios[-1]:.6f}"

    # Clear old text and redraw
    [t.remove() for t in ax.texts]
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    plt.pause(0.01)  # refresh the figure


def evaluate_saved_model(saved_filename):
    """Load a saved model and run MIP analysis on it"""

    # Load the saved model
    model_path = f"models/{saved_filename}"
    target_network = torch.load(model_path, map_location=device, weights_only=False)
    
    functional_state_dict = target_network.state_dict()
    print("\n=== FunctionalR Parameters ===")
    for param_name, param_tensor in functional_state_dict.items():
        print(f"\nParameter: {param_name}")
        print(f"Shape: {param_tensor.shape}")
        print(f"Device: {param_tensor.device}")
        print(f"Dtype: {param_tensor.dtype}")
        print(f"Values:\n{param_tensor}")
        print(f"Min: {param_tensor.min().item()}, Max: {param_tensor.max().item()}")
        print("-" * 50)

    print(f"Loaded model: {saved_filename}")
    print(f"Model's achievable_alpha_delta: {target_network.achievable_alpha_delta}")

    # Run MIP analysis
    alpha_delta = 0

    print(f"Running MIP analysis with alpha_delta={alpha_delta}")

    (
        wcp_left,
        wcp_right,
        error_left,
        error_right,
        total_error,
    ) = target_network.worst_case_analysis(alpha_delta=alpha_delta)

    current_ratio = alpha - total_error

    print(f"Results:")
    print(f"  Worst case profile left: {wcp_left}")
    print(f"  Worst case profile right: {wcp_right}")
    print(f"  Error left: {error_left:.20f}")
    print(f"  Error right: {error_right:.20f}")
    print(f"  Total error: {total_error:.20f}")
    print(f"  Current allocative ratio: {current_ratio:.20f}")

    return current_ratio

class KeyboardListener:
    def __init__(self):
        self.stop_training = False
        self.listening = False

    def start_listening(self):
        """Start listening for keyboard input in a separate thread."""
        self.listening = True

        def listen():
            while not self.stop_training and self.listening:
                if msvcrt.kbhit():  # check if a key was pressed
                    char = msvcrt.getch().decode("utf-8", errors="ignore")
                    if char.lower() == "i":
                        print("\n🛑 Manual stop requested by pressing 'i'")
                        self.stop_training = True
                        break
                time.sleep(0.1)  # small sleep to avoid busy waiting

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def cleanup(self):
        """Stop listening (no terminal settings to restore on Windows)."""
        self.listening = False
            
# Jul_31_15hr_40min_37sec-5-3-0-2-03995-0.00005244780750190969.saved
# evaluate_saved_model("Jun_09_15hr_50min_08sec-5-3-0-2-07466-0.00000751132112641884.saved")         
# evaluate_saved_model("Jul_31_15hr_40min_37sec-5-3-0-2-03995-0.00005244780750190969.saved")             

# run training from scratch or resume from a checkpoint
if resume:
    pop,gen,wcp = load_checkpoint_with_info()
    run_training(time_limit*60, epochs_before_analysis, init_alpha_delta, pop, wcp)
else:
    run_training(time_limit*60, epochs_before_analysis, init_alpha_delta)

"""
Order of execution:
1. calls hypernetwork()
2. run_hypernetwork_with_architecture() to get max_ratio
3. hypernetwork_train()
4. plot_allocative_ratios()
5. evaluate_saved_model()

NEEDS TO BE

1. init neat
2. test on victor profiles
3. worst-case analysis via mip
4. repopulate
5. repeat
"""