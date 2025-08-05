import os
import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from hypernetwork import Hypernetwork
from itertools import count
from pprint import pprint
from settings import (
    n,
    finger_print,
    device,
    env_alpha_delta,
    resume_filename,
)

# from model import R, REnsemble  # mutate, crossover
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
from model import FunctionalR
import threading
import sys
import select
import tty
import termios


def format_elapsed_time(start_time):
    """Calculate and format elapsed time as HH:MM:SS or HH_MM_SS format"""
    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    return elapsed_time, f"{elapsed_hours:02d}h:{elapsed_minutes:02d}m:{elapsed_seconds:02d}s", f"{elapsed_hours:02d}h_{elapsed_minutes:02d}m_{elapsed_seconds:02d}s"


def hypernetwork():
    target_architectures = [
        [6, 20, 1],
    ]

    hypernetwork_configs = [
        [1024, 512, 256],
    ]

    results = {}
    results_file = f"architecture_comparison_n=6_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    for target_arch in target_architectures:
        for hn_config in hypernetwork_configs:
            print(f"\n\n{'='*60}")
            print(f"Testing target: {target_arch}, hypernetwork: {hn_config}")
            print(f"{'='*60}\n")

            max_ratio = run_hypernetwork_with_architecture(
                target_arch, hn_config, time_limit_minutes=18000
            )
            
            arch_key = f"target_{'-'.join(map(str, target_arch))}_hn_{'-'.join(map(str, hn_config))}"
            results[arch_key] = max_ratio

            with open(results_file, "a") as f:
                f.write(f"{arch_key}: {max_ratio}\n")

            print(f"Result for {arch_key}: {max_ratio}")


def run_hypernetwork_with_architecture(target_arch, hn_config, time_limit_minutes):
    time_limit_seconds = time_limit_minutes * 60
    
    # Initialize the target network and hypernetwork
    target_network = FunctionalR(target_arch).to(device)

    hypernetwork = Hypernetwork(
        hidden_layers=hn_config,
        activation="relu",
        target_network=target_network,
    ).to(device)
    
    target_params = sum(p.numel() for p in target_network.parameters() if p.requires_grad)
    hn_params = sum(p.numel() for p in hypernetwork.parameters() if p.requires_grad)
    print(f"Target network parameters: {target_params}")
    print(f"Hypernetwork parameters: {hn_params}")
    print(f"Parameter ratio (HN/Target): {hn_params / target_params:.4f}")
    #############################
    
    optimizer = optim.Adam(hypernetwork.parameters(), lr=0.001)
    
    worst_case_profiles = []
    total_error_history = []
    
    # Get alpha delta
    if env_alpha_delta == 0:
        achievable_alpha_delta = 0
    else:
        if resume_filename is None:
            achievable_alpha_delta = manual_error
        else:
            achievable_alpha_delta = target_network.achievable_alpha_delta
            print(f"restored achievable_alpha_delta to {achievable_alpha_delta}")
    alpha_delta = max(achievable_alpha_delta - 0.001, 0)
    #############################
    print(
        f"hypernetwork {finger_print} alpha_delta={alpha_delta}"
    )
    
    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()
    
    try:
        max_allocative_ratio = 0.0
        start_time = time.time()
        
        allocative_ratios = []
        iterations = []
        
        time_of_last_max_improvement = start_time
        
        
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
            
            if keyboard_listener.stop_training:
                print("Training stopped by user input.")
                break
            
            profiles = victor_profiles + worst_case_profiles[-256:]
            
            # Train the hypernetwork
            training_start_time = time.time()
            avg_loss = hypernetwork_train(
                hypernetwork,
                target_network,
                optimizer,
                profiles,
                alpha_delta=alpha_delta,
                device=device,
                epochs=500,
            )
            training_duration = time.time() - training_start_time
            #############################
            
            # Running worst case analysis
            worst_case_start_time = time.time()
            (
                wcp_left,
                wcp_right,
                error_left,
                error_right,
                total_error,
            ) = target_network.worst_case_analysis(alpha_delta=alpha_delta)
            worst_case_duration = time.time() - worst_case_start_time
            #############################
            
            # Add worst case profiles
            worst_case_profiles = add_two_profiles(
                wcp_left, wcp_right, worst_case_profiles
            )
            iteration_duration = time.time() - iteration_start_time   
            ############################# 
            
            # Save model 
            should_save_model = total_error < min(total_error_history, default=(1000, 0))[0]
            if should_save_model:
                target_network.achievable_alpha_delta = achievable_alpha_delta
                fn = f"saved/{finger_print}-{iter_index:05}-{total_error:.20f}.saved"
                # Ensure the directory exists [trungtruong]
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                torch.save(target_network, fn)

            total_error_history.append((total_error, iter_index))
            achievable_alpha_delta = min(achievable_alpha_delta, total_error)

            if total_error - alpha_delta < 0.001:
                new_alpha_delta = max(alpha_delta / 2, alpha_delta - 0.01)
            else:
                new_alpha_delta = (alpha_delta + achievable_alpha_delta) / 2
            new_alpha_delta = max(
                min(new_alpha_delta, achievable_alpha_delta - 0.001), 0)
            print(f"alpha_delta goes from {alpha_delta} to {new_alpha_delta}")
            alpha_delta = new_alpha_delta
            #############################
            
            # Calculate elapsed time from the very start
            elapsed_time, elapsed_time_str, elapsed_time_underscore = format_elapsed_time(start_time)
            
            # Check current allocative ratio vs max allocative ratio
            current_ratio = alpha - total_error
            
            # Save hypernetwork model
            if should_save_model:
                os.makedirs("hypernets", exist_ok=True)
                timestamp = time.strftime("%b_%d_%Hhr_%Mmin_%Ssec")
                hn_filename = f"hypernets/{timestamp}-n{n}-ratio{current_ratio:.8f}-time{elapsed_time_str}-{finger_print}.hn"
                
                save_data = {
                    'hypernetwork_state_dict': hypernetwork.state_dict(),
                    'max_allocative_ratio': current_ratio,
                    'elapsed_time_seconds': elapsed_time,
                    'elapsed_time_formatted': elapsed_time_str,
                    'n': n,
                    'finger_print': finger_print,
                    'target_arch': target_arch,
                    'hn_config': hn_config,
                    'timestamp': timestamp,
                    'iteration': iter_index,
                    'total_error': total_error
                }
                
                torch.save(save_data, hn_filename)
            
            if current_ratio > max_allocative_ratio:
                time_of_last_max_improvement = time.time()
                max_allocative_ratio = current_ratio
            time_since_improvement = time.time() - time_of_last_max_improvement
            hours_since = int(time_since_improvement // 3600)
            minutes_since = int((time_since_improvement % 3600) // 60)
            seconds_since = int(time_since_improvement % 60)
            time_since_str = f"{hours_since:02d}h:{minutes_since:02d}m:{seconds_since:02d}s"
            #############################
            
            # Check if no improvement for more than 2 hours
            if time_since_improvement > 20 * 1800:  # 2 hours in seconds
                print(f"⏰ No improvement in max allocative ratio for {time_since_str}. Stopping training.")
                break
            #############################
            allocative_ratios.append(current_ratio)
            iterations.append(iter_index)
            
            print(
                f"elapsed_time={elapsed_time_str} "
                f"time_since_max_improvement={time_since_str} "
                # f"error_left={error_left:.20f} "
                # f"error_right={error_right:.20f} "
                f"alpha_delta={alpha_delta:.20f} "
                f"final_loss={avg_loss:.20f} "
                f"max_allocative_ratio={max_allocative_ratio:.20f} "
                f"training_time={training_duration:.2f}s "
                f"worst_case_time={worst_case_duration:.2f}s "
                f"iteration_time={iteration_duration:.2f}s"
            )
            print(f"Current worst case allocative ratio: {current_ratio}")
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
                break
        
        plot_allocative_ratios(iterations, allocative_ratios, target_arch, hn_config)
    
        return max_allocative_ratio
    
    finally:
        keyboard_listener.cleanup()

def hypernetwork_train(
    hypernetwork,
    target_network,
    optimizer,
    profiles,
    alpha_delta,
    device,
    worst_case_profiles=None,
    epochs=100,
):
    # hypernetwork = hypernetwork.to(device)
    # target_network = target_network.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Generate flat parameters from the hypernetwork (optimized approach)
        flat_params = hypernetwork.get_flat_parameters()
        target_network.update_params_flat(flat_params)
        
        profiles_all = profiles + get_random_profiles(256)
        
        # profiles_tensor = torch.tensor(
        #     vectorized_kick_batch(profiles_all),
        #     dtype=torch.float32,
        #     device=device,
        # )
        
        profiles_np = vectorized_kick_batch(profiles_all) 
        profiles_tensor = torch.from_numpy(profiles_np).float().to(device)

        # s_tensor = torch.tensor(
        #     s_batch(profiles_all),
        #     dtype=torch.float32,
        #     device=device,
        # )
        
        s_np = s_batch(profiles_all)  
        s_tensor = torch.from_numpy(s_np).float().to(device)
        
        total_r = torch.sum(target_network(profiles_tensor), dim=1)

        loss = torch.sum(
            torch.square(
                torch.clamp(total_r - (n - (alpha - alpha_delta)) * s_tensor, min=0)
            )
        ) + torch.sum(torch.square(torch.clamp((n - 1) * s_tensor - total_r, min=0)))
        current_loss = loss.item()
        
        # If solved this batch of profiles, break early
        if math.isclose(current_loss, 0.0, abs_tol=0.0):
            break
        #############################

        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_norm=1.0)

        optimizer.step()
        
    return current_loss

def plot_allocative_ratios(iterations, ratios, target_arch, hn_config):
    """Plot allocative efficiency ratios over iterations"""

    plt.figure(figsize=(12, 8))

    # Main plot
    plt.plot(iterations, ratios, "b-", linewidth=2, label="Current Ratio")

    # Add max achieved line
    max_ratio = max(ratios)
    plt.axhline(
        y=max_ratio,
        color="g",
        linestyle=":",
        linewidth=1,
        label=f"Max Achieved ({max_ratio:.6f})",
    )

    # Formatting
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Allocative Efficiency Ratio", fontsize=12)
    plt.title(
        f"Allocative Efficiency Progress\nTarget: {target_arch}, Hypernetwork: {hn_config}",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set y-axis limits for better visibility
    min_ratio = min(ratios)
    plt.ylim(max(0, min_ratio - 0.01), min(1, max_ratio + 0.01))

    # Add statistics text box
    stats_text = (
        f"Iterations: {len(iterations)}\nMax: {max_ratio:.6f}\nFinal: {ratios[-1]:.6f}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"plots/allocative_ratio_plot_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {filename}")

    plt.show()
    
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
        self.old_settings = None
        self.listening = False

    def start_listening(self):
        """Start listening for keyboard input in a separate thread"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            self.listening = True
            
            def listen():
                # Don't set raw mode - use cbreak mode instead
                tty.setcbreak(sys.stdin.fileno())
                
                while not self.stop_training and self.listening:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        try:
                            char = sys.stdin.read(1)
                            if char.lower() == "i":
                                print("\n🛑 Manual stop requested by pressing 'i'")
                                self.stop_training = True
                                break
                        except:
                            break

            thread = threading.Thread(target=listen, daemon=True)
            thread.start()
        except:
            # If terminal setup fails, just continue without keyboard listener
            print("Warning: Could not set up keyboard listener")

    def cleanup(self):
        """Restore terminal settings"""
        self.listening = False
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
            
# Jul_31_15hr_40min_37sec-5-3-0-2-03995-0.00005244780750190969.saved
# evaluate_saved_model("Jun_09_15hr_50min_08sec-5-3-0-2-07466-0.00000751132112641884.saved")         
# evaluate_saved_model("Jul_31_15hr_40min_37sec-5-3-0-2-03995-0.00005244780750190969.saved")             
hypernetwork()