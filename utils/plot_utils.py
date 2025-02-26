import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

def plot_normalized_data(data, normalized_data, sample_fraction, output_dir, current_date):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], normalized_data, label='Normalized Data')
    plt.title(f'Step 1: Normalized HKHS Index (sample_fraction={sample_fraction})')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_dir, f"{current_date}", f"{sample_fraction}_Step1_Normalized_HKHS_Index.png")
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_significant_points(data, normalized_data, significant_indices,output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], normalized_data, label='Normalized Data')
    plt.scatter(data['date'].iloc[significant_indices], normalized_data[significant_indices],
                color='red', label='Significant Points', marker='x')
    plt.title('Step 2: Significant Points Sampling')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_dir, f"{current_date}", f"{sample_fraction}_Step2_Significant_Points_Sampling.png")
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_budget_allocation(budgets, output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(budgets)), budgets, color='blue')
    plt.title('Step 3: Adaptive Budget Allocation with w-event Privacy')
    plt.xlabel('Significant Point Index')
    plt.ylabel('Privacy Budget (epsilon)')
    plt.grid(True)
    file_path = os.path.join(output_dir, f"{current_date}", f"{sample_fraction}_Step3_Budget_Allocation.png")
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'].iloc[significant_indices], perturbed_values,
         label='Perturbed Points', color='red', linestyle='--', marker='x')
    plt.plot(data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1)
    plt.title('Step 4: SW Mechanism Perturbation')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_dir, f"{current_date}", f"{sample_fraction}_Step4_Perturbation.png")
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1, 
         zorder=3, marker='.', markersize=3)
    plt.scatter(data['date'].iloc[significant_indices], perturbed_values,
            label='Perturbed Points', color='red', marker='x')
    plt.plot(data['date'].iloc[significant_indices], smoothed_values,
         label='Smoothed Points', color='blue', linestyle='-', linewidth=1, zorder=1)
    plt.title('Step 5: Kalman Filter Smoothing')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_dir, f"{current_date}", f"{sample_fraction}_Step5_Kalman_Smoothing.png")
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(file_path, dpi=300)
    plt.close()
