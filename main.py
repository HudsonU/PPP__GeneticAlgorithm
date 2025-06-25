from typing import Optional, Iterable, Dict, Any, Tuple
import torch.nn as nn
import os
import torch
import time
import torch.optim as optim
import numpy as np
import timeit
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from math import ceil
from hypernetwork import Hypernetwork
from itertools import count
from random import sample  # , randrange
from pprint import pprint
from settings import (
    n,
    story,
    env_shapes,
    finger_print,
    # env_size,
    device,
    env_alpha_delta,
    env_train_using_worst_case,
    env_train_using_free_samples,
    resume_filename,
    story3_worst_case_done_limit,
)

# from model import R, REnsemble  # mutate, crossover
from utility import (
    loss_schedule,
    alpha,
    victor_profiles,
    manual_error,
    s,
    kick,
    vectorized_kick,
    add_two_profiles,
    get_random_profiles,
)
from model import FunctionalR

import threading
import sys
import select
import tty
import termios
# [trungtruong]


def hypernetwork():
    # Test different target network architectures
    target_architectures = [
        [3, 5, 1],
    ]

    hypernetwork_configs = [
        [128, 64, 32],
    ]

    # results = {}
    # results_file = f"architecture_comparison_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    for target_arch in target_architectures:
        for hn_config in hypernetwork_configs:
            print(f"\n\n{'='*60}")
            print(f"Testing target: {target_arch}, hypernetwork: {hn_config}")
            print(f"{'='*60}\n")

            max_ratio = run_hypernetwork_with_architecture(
                target_arch, hn_config, time_limit_minutes=900
            )

            # arch_key = f"target_{'-'.join(map(str, target_arch))}_hn_{'-'.join(map(str, hn_config))}"
            # results[arch_key] = max_ratio

            # with open(results_file, "a") as f:
            #     f.write(f"{arch_key}: {max_ratio}\n")

            # print(f"Result for {arch_key}: {max_ratio}")


def run_hypernetwork_with_architecture(target_arch, hn_config, time_limit_minutes=10):
    target_network = FunctionalR(target_arch).to(device)
    print(f"Target network architecture: {target_arch}")

    # Print architecture info
    total_params = 0
    for layer in target_network.get_parameter_shapes():
        weight_shape, bias_shape = layer
        layer_params = np.prod(weight_shape) + np.prod(bias_shape)
        total_params += layer_params
    print(f"Total target parameters: {total_params}")

    hypernetwork = Hypernetwork(
        hidden_layers=hn_config,
        activation="relu",
        target_network=target_network,
    ).to(device)
    hn_params = sum(p.numel() for p in hypernetwork.parameters() if p.requires_grad)
    print(f"Hypernetwork parameters: {hn_params}")
    
    print(f"Parameter ratio (HN/Target): {hn_params/total_params:.2f}")

    optimizer = optim.Adam(hypernetwork.parameters(), lr=0.001)

    worst_case_profiles = []
    # total_error_history = []
    if env_alpha_delta == 0:
        achievable_alpha_delta = 0
    else:
        if resume_filename is None:
            achievable_alpha_delta = manual_error
        else:
            achievable_alpha_delta = target_network.achievable_alpha_delta
            print(f"restored achievable_alpha_delta to {achievable_alpha_delta}")
            
    print(f"achievable_alpha_delta={achievable_alpha_delta}")
    alpha_delta = max(achievable_alpha_delta - 0.001, 0)
    train_using_worst_case = env_train_using_worst_case == 1
    train_using_free_samples = env_train_using_free_samples == 1
    print(
        f"hypernetwork {finger_print} alpha_delta={alpha_delta}, train_using_worst_case={train_using_worst_case}, train_using_free_samples={train_using_free_samples}"
    )
    # train_count = 0
    # train_count_limit = 10
    # worst_case_done = 0

    # Initialize keyboard listener
    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()

    try:
        # Track the maximum allocative ratio achieved
        max_allocative_ratio = 0.0
        
        # Track the time 
        start_time = time.time()
        time_limit_seconds = time_limit_minutes * 60        
        
        # Track allocative ratios for plotting
        allocative_ratios = []
        iterations = []
        
        # Track time since max ratio improvement
        time_of_last_max_improvement = start_time

        for iter_index in count(start=1):
            # Start timing the entire iteration
            iteration_start_time = time.time()

            # Check stopping conditions
            # elapsed_time_check = time.time() - start_time
            # if elapsed_time_check >= time_limit_seconds:
            #     print(
            #         f"Time limit of {time_limit_minutes} minutes reached. Stopping training."
            #     )
            #     break

            if keyboard_listener.stop_training:
                print("Training stopped by user input.")
                break

            # if worst_case_done == story3_worst_case_done_limit:
            #     break

            profiles = victor_profiles + worst_case_profiles[-256:]  # noqa: E203

            # print(f"Training with {len(profiles) + 256} profiles")

            # Start timing the training
            training_start_time = time.time()
            avg_loss = hypernetwork_train(
                hypernetwork,
                target_network,
                optimizer,
                profiles,
                alpha_delta=alpha_delta,
                device=device,
                # worst_case_profiles=(
                #     worst_case_profiles[:-16] if train_using_worst_case else None
                # ),
                # worst_case_profiles=(
                #     worst_case_profiles if train_using_worst_case else None
                # ),
                epochs=300,
            )
            training_duration = time.time() - training_start_time
            # train_count += 1
            # if train_using_free_samples:
            #     print(
            #         f"iter={iter_index} avg_loss={avg_loss:.20f} loss_schedule={loss_schedule(train_count):.20f} with train_count {train_count}"
            #     )
            #     if (
            #         avg_loss > loss_schedule(train_count)
            #         and train_count < train_count_limit
            #     ):
            #         continue
            #     else:
            #         train_count = 0
            worst_case_start_time = time.time()
            (
                wcp_left,
                wcp_right,
                error_left,
                error_right,
                total_error,
            ) = target_network.worst_case_analysis(alpha_delta=alpha_delta)
            worst_case_duration = time.time() - worst_case_start_time
            # worst_case_done += 1
            worst_case_profiles = add_two_profiles(
                wcp_left, wcp_right, worst_case_profiles
            )            
            
            # Calculate total iteration time
            iteration_duration = time.time() - iteration_start_time

            # Calculate elapsed time from start (at end of iteration)
            elapsed_time = time.time() - start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_seconds = int(elapsed_time % 60)
            elapsed_time_str = (
                f"{elapsed_hours:02d}h:{elapsed_minutes:02d}m:{elapsed_seconds:02d}s"
            )

            current_ratio = alpha - total_error
              
            # Check if max allocative ratio improved and update time tracking
            if current_ratio > max_allocative_ratio:
                time_of_last_max_improvement = time.time()
                max_allocative_ratio = current_ratio
            
            # Calculate time since last max improvement
            time_since_improvement = time.time() - time_of_last_max_improvement
            hours_since = int(time_since_improvement // 3600)
            minutes_since = int((time_since_improvement % 3600) // 60)
            seconds_since = int(time_since_improvement % 60)
            time_since_str = f"{hours_since:02d}h:{minutes_since:02d}m:{seconds_since:02d}s"

            # Check if no improvement for more than 2 hours
            if time_since_improvement > 2 * 3600:  # 2 hours in seconds
                print(f"No improvement in max allocative ratio for {time_since_str}. Stopping training.")
                break

            # Check if we've reached near-optimal allocative efficiency
            optimal_ratio = 0.868421 
            if current_ratio >= optimal_ratio - 0.0001:
                print(
                    f"NEAR-OPTIMAL ACHIEVED! Current ratio: {current_ratio:.8f}, Target: {optimal_ratio:.8f}"
                )
                print("Stopping training as we've reached near-optimal performance!")
                break            
            
            # Store data for plotting
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

        # Plot the results
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
    hypernetwork = hypernetwork.to(device)
    target_network = target_network.to(device)

    # total_loss = 0
    # losses = []
    grad_norms = []
    param_changes = []

    # Store initial hypernetwork parameters for comparison
    initial_params = [p.clone().detach() for p in hypernetwork.parameters()]

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate weights from the hypernetwork
        params = hypernetwork.get_parameters_for_target()
        params = [(w.to(device), b.to(device)) for w, b in params]
        target_network.update_params(params)

        # Increase sample diversity significantly
        if worst_case_profiles is None:
            profiles_all = profiles + get_random_profiles(256)
        else:
            profiles_all = (
                profiles
                # + sample(
                #     worst_case_profiles,
                #     min(256, len(worst_case_profiles)),
                # )
                + get_random_profiles(256)
            )

        profiles_tensor = torch.tensor(
            [vectorized_kick(profile) for profile in profiles_all],
            dtype=torch.float32,
            device=device,
        )
        s_tensor = torch.tensor(
            [[s(profile)] for profile in profiles_all],
            dtype=torch.float32,
            device=device,
        )
        total_r = torch.sum(target_network(profiles_tensor), dim=1)

        loss = torch.sum(
            torch.square(
                torch.clamp(total_r - (n - (alpha - alpha_delta)) * s_tensor, min=0)
            )
        ) + torch.sum(torch.square(torch.clamp((n - 1) * s_tensor - total_r, min=0)))

        loss.backward()

        # Monitor gradient norms
        total_grad_norm = 0
        for p in hypernetwork.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm**0.5
        grad_norms.append(total_grad_norm)

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_norm=1.0)

        optimizer.step()

        current_loss = loss.item()
        # Check if loss is exactly zero
        if math.isclose(current_loss, 0.0, abs_tol=0.0):
            # print("Loss is exactly zero, stopping training.")
            break
        # losses.append(current_loss)
        # total_loss += current_loss

        # Monitor parameter changes
        # if epoch % 50 == 0:
        #     param_change = 0
        #     for initial, current in zip(initial_params, hypernetwork.parameters()):
        #         param_change += (current - initial).norm().item()
        #     param_changes.append(param_change)

        #     # Monitor parameter changes
        #     if epoch % 50 == 0:
        #         param_change = 0
        #         for initial, current in zip(initial_params, hypernetwork.parameters()):
        #             param_change += (current - initial).norm().item()
        #             param_changes.append(param_change)
        #             print(
        #             f"  Epoch {epoch}: Loss={current_loss:.6f}, GradNorm={total_grad_norm:.6f}, ParamChange={param_change:.6f}, AlphaDelta={alpha_delta:.6f}"
        #             )

    # Print training summary
    # print(
    #     f"Training summary: Final loss={losses[-1]:.6f}, Avg grad norm={np.mean(grad_norms):.6f}"
    # )
    # if len(grad_norms) > 0 and np.mean(grad_norms) < 1e-6:
    #     print("WARNING: Very small gradients - potential vanishing gradient problem!")

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
    model_path = f"saved/{saved_filename}"
    target_network = torch.load(model_path, map_location=device, weights_only=False)

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


# Example usage:
# ratio = evaluate_saved_model("Jun_09_15hr_50min_08sec-5-3-0-2-07466-0.00000751132112641884.saved")
# story3_hypernetwork()
# story3()
hypernetwork()
