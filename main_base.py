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


def hypernetwork():
    target_arch = [5, 20, 1]
    hn_config = [768, 384, 192]

    run_hypernetwork_with_architecture(target_arch, hn_config)


def run_hypernetwork_with_architecture(target_arch, hn_config):
    
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
    
    train_using_worst_case = env_train_using_worst_case == 1
    train_using_free_samples = env_train_using_free_samples == 1
    print(
        f"hypernetwork {finger_print} alpha_delta={alpha_delta}, train_using_worst_case={train_using_worst_case}, train_using_free_samples={train_using_free_samples}"
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
            
            if keyboard_listener.stop_training:
                print("Training stopped by user input.")
                break
            
            profiles = victor_profiles + worst_case_profiles[-32:]
            
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
            
            # Alpha delta adjustment
            if total_error < min(total_error_history, default=(1000, 0))[0]:
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
            elapsed_time = time.time() - start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_seconds = int(elapsed_time % 60)
            elapsed_time_str = (
                f"{elapsed_hours:02d}h:{elapsed_minutes:02d}m:{elapsed_seconds:02d}s"
            )
            
            # Check current allocative ratio vs max allocative ratio
            current_ratio = alpha - total_error
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
            if time_since_improvement > 2 * 3600:  # 2 hours in seconds
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

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Generate weights from the hypernetwork
        params = hypernetwork.get_parameters_for_target()
        params = [(w.to(device), b.to(device)) for w, b in params]
        target_network.update_params(params)
        
        profiles_all = profiles + get_random_profiles(32)
        
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
        current_loss = loss.item()
        
        # If solved this batch of profiles, break early
        if math.isclose(current_loss, 0.0, abs_tol=0.0):
            break
        #############################

        loss.backward()
        
        # Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_norm=1.0)

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
            
hypernetwork()