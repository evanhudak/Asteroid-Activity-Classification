import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Function to plot a histogram with count labels on each bar
def plot_histogram(data, bins, xlabel, ylabel, title):
    # Plot histogram of given data with specified bins
    n, bins, patches = plt.hist(data, bins=bins)

    # Add text labels above each histogram bar showing count
    for count, edge_left, patch in zip(n, bins, patches):
        x = patch.get_x() + patch.get_width() / 2    # Center x-position of bar
        y = patch.get_height()                       # Height of bar
        plt.text(x, y + 0.5, str(int(count)), ha='center', va='bottom', fontsize=9)
    
    # Set x-axis ticks, labels, and title
    plt.xticks(bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display the plot    
    plt.show()

# Function to plot a cumulative histogram (CDF)
def plot_cumulative_distribution(data, title):
    # Set x- and y-axis labels and plot title
    plt.xlabel("# of Days From First True to Last True")
    plt.ylabel("%")
    plt.title(title)

    # Define bin edges from 0 to 364 (1-day intervals)
    bins = np.arange(0, 365, 1)

    # Plot cumulative histogram as a step function
    plt.hist(data, bins=bins, density=True, cumulative=True, histtype="step")
    
    # Display the plot
    plt.show()

# Main analysis function (no database access required)
def analyze_asteroid_group(json_file, group_name):
    # Load asteroid prediction data from JSON file
    with open(json_file, 'r') as f:
        asteroid_data = json.load(f)

    # Initialize containers
    interesting_ast = set()          # Asteroids with long streaks of "True"
    counter_data = []                # List of all consecutive True counts
    max_counter_data = []            # Max streak per asteroid
    time_apart = []                  # Duration from first to last True in a streak
    ast_count = {}                   # Map from asteroid ID to max streak

    # Iterate through each asteroid's predictions
    for asteroid in asteroid_data:
        ast_num = asteroid["AsteroidNum"]
        detections = asteroid["detections"]

        # Skip if no detections
        if not detections:
            continue

        # Sort detections by jd
        detections.sort(key=lambda x: x["jd"])

        # Initialize counters for tracking streaks
        counter = 0
        max_count = 0
        first_true = 0
        longest_time = 0

        for det in detections:
            if det["predicted"] == "True":
                # Increment counter for consecutive "True" values
                counter += 1
                if counter == 1:
                    first_true = det["jd"]  # Record start of the streak
                if counter >= 20:
                    interesting_ast.add(ast_num)  # Mark as interesting
            else:
                # Record the streak
                counter_data.append(counter)

                # If a long enough streak, calculate duration
                if counter >= 7 and first_true != 0:
                    days = det["jd"] - first_true
                    if 1 < days < 365 and days > longest_time:
                        longest_time = days
                        time_apart.append(longest_time)

                # Update max streak and reset counter
                max_count = max(max_count, counter)
                counter = 0

        # Handle case where final detection is part of a streak
        counter_data.append(counter)
        max_count = max(max_count, counter)
        if counter >= 7 and first_true != 0:
            days = detections[-1]["jd"] - first_true
            if 1 < days < 365 and days > longest_time:
                longest_time = days
                time_apart.append(longest_time)

        max_counter_data.append(max_count)
        ast_count[ast_num] = max_count

    # ---- Plotting Results ----

    # Plot histogram of max consecutive "True" counts
    plot_histogram(max_counter_data, bins=np.arange(0, 75, 10),
                   xlabel="# of Consecutive Trues",
                   ylabel="#",
                   title=f"Histogram for Consecutive Trues in {group_name}")

    # Plot histogram of time durations from first to last "True"
    plot_histogram(time_apart, bins=np.arange(0, 365, 30),
                   xlabel="# of Days",
                   ylabel="#",
                   title=f"Histogram for Days of Consecutive Trues in {group_name}")

    # Plot cumulative distribution of durations
    plot_cumulative_distribution(time_apart,
                                 title=f"Cumulative Distribution Plot for {group_name}")

    # Return the original JSON DataFrame
    return asteroid_data