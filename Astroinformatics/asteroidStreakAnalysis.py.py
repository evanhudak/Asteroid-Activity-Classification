import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Connect to MongoDB using specified credentials and server
pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

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

# Function to analyze asteroid predictions from a JSON file and plot statistics
def analyze_asteroid_group(json_file, group_name):
    # Load asteroid prediction data from JSON file
    data = pd.read_json(json_file)
    
    # Set the asteroid number as the index
    data = data.set_index('AsteroidNum')

    # Initialize containers for results
    interesting_ast = set()           # Asteroids with long streaks of "True"
    counter_data = []                 # List of all consecutive True counts
    max_counter_data = []            # Max streak per asteroid
    time_apart = []                  # Duration from first to last True in a streak
    ast_count = {}                   # Map from asteroid ID to max streak

    # Iterate through each asteroid's predictions
    for index, row in data.iterrows():
        print(f"Starting for asteroid {index}")
        ZTFID_list = []              # List of all detection IDs
        ZTFID_predictions = {}       # Mapping from ID to predicted value

        # Query MongoDB for all observations of this asteroid
        ssid_data = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': index}))
        
        # Add 'Predicted' column if missing
        if 'Predicted' not in ssid_data.columns:
            ssid_data['Predicted'] = None

        # Extract prediction info from the JSON file
        for pred in row:
            for detect in pred:
                ZTFID_list.append(detect['id'])
                ZTFID_predictions[detect['id']] = detect['predicted']

        # Keep only detections that exist in prediction list and assign prediction values
        for ind, row in ssid_data.iterrows():
            if ssid_data.loc[ind, 'id'] not in ZTFID_list:
                ssid_data = ssid_data.drop(ind)
            else:
                ssid_data.loc[ind, 'Predicted'] = ZTFID_predictions[ssid_data.loc[ind, 'id']]

        # Reset index after dropping rows
        ssid_data.reset_index(drop=True, inplace=True)

        # Initialize counters for tracking streaks
        counter = 0
        max_count = 0
        longest_time = 0
        first_true = 0

        # Iterate over rows sorted by Julian Date (jd)
        for _, row in ssid_data.sort_values(by='jd').iterrows():
            if row["Predicted"] == "True":
                # Increment counter for consecutive "True" values
                counter += 1
                if counter == 1:
                    first_true = row["jd"]  # Record start of the streak
                if counter >= 20:
                    interesting_ast.add(row['ssnamenr'])  # Mark as interesting
            else:
                # End of a streak; record the count
                counter_data.append(counter)

                # If a long enough streak, calculate duration
                if counter >= 7 and first_true != 0:
                    days = row["jd"] - first_true
                    if 1 < days < 365 and days > longest_time:
                        longest_time = days
                        time_apart.append(longest_time)
                
                # Update max streak and reset counter
                max_count = max(max_count, counter)
                counter = 0

        # Save the max streak length for this asteroid
        max_counter_data.append(max_count)
        ast_count[index] = max_count

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
    return data