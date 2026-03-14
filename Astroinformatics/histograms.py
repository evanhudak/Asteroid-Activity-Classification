import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

def plot_histogram(data, bins, xlabel, ylabel, title):
    n, bins, patches = plt.hist(data, bins=bins)
    for count, edge_left, patch in zip(n, bins, patches):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y + 0.5, str(int(count)), ha='center', va='bottom', fontsize=9)
    plt.xticks(bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_cumulative_distribution(data, title):
    plt.xlabel("# of Days From First True to Last True")
    plt.ylabel("%")
    plt.title(title)
    bins = np.arange(0, 365, 1)
    plt.hist(data, bins=bins, density=True, cumulative=True, histtype="step")
    plt.show()

def analyze_asteroid_group(json_file, group_name):
    data = pd.read_json(json_file)
    data = data.set_index('AsteroidNum')

    interesting_ast = set()
    counter_data = []
    max_counter_data = []
    time_apart = []
    ast_count = {}

    for index, row in data.iterrows():
        print(f"Starting for asteroid {index}")
        ZTFID_list = []
        ZTFID_predictions = {}

        ssid_data = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': index}))
        if 'Predicted' not in ssid_data.columns:
            ssid_data['Predicted'] = None

        for pred in row:
            for detect in pred:
                ZTFID_list.append(detect['id'])
                ZTFID_predictions[detect['id']] = detect['predicted']

        for ind, row in ssid_data.iterrows():
            if ssid_data.loc[ind, 'id'] not in ZTFID_list:
                ssid_data = ssid_data.drop(ind)
            else:
                ssid_data.loc[ind, 'Predicted'] = ZTFID_predictions[ssid_data.loc[ind, 'id']]

        ssid_data.reset_index(drop=True, inplace=True)

        counter = 0
        max_count = 0
        longest_time = 0
        first_true = 0

        for _, row in ssid_data.sort_values(by='jd').iterrows():
            if row["Predicted"] == "True":
                counter += 1
                if counter == 1:
                    first_true = row["jd"]
                if counter >= 20:
                    interesting_ast.add(row['ssnamenr'])
            else:
                counter_data.append(counter)
                if counter >= 7 and first_true != 0:
                    days = row["jd"] - first_true
                    if 1 < days < 365 and days > longest_time:
                        longest_time = days
                        time_apart.append(longest_time)
                max_count = max(max_count, counter)
                counter = 0

        max_counter_data.append(max_count)
        ast_count[index] = max_count

    # Plotting
    plot_histogram(max_counter_data, bins=np.arange(0, 75, 10),
                   xlabel="# of Consecutive Trues",
                   ylabel="#",
                   title=f"Histogram for Consecutive Trues in {group_name}")

    plot_histogram(time_apart, bins=np.arange(0, 365, 30),
                   xlabel="# of Days",
                   ylabel="#",
                   title=f"Histogram for Days of Consecutive Trues in {group_name}")

    plot_cumulative_distribution(time_apart,
                                 title=f"Cumulative Distribution Plot for {group_name}")

    return data

def find_consecutive_true_jd_differences_hist(asteroid_number, min_consecutive, json_file):
    with open(json_file, 'r') as f:
        asteroid_data = json.load(f)

    if not isinstance(asteroid_data, list):
        asteroid_data = [asteroid_data]

    ztf_predictions = set()
    for asteroid in asteroid_data:
        if int(asteroid.get("AsteroidNum")) == asteroid_number:
            for detection in asteroid.get("detections", []):
                if detection.get("predicted", "").strip().lower() == "true":
                    ztf_predictions.add(detection.get("id"))
            break

    if not ztf_predictions:
        print(f"No predicted True values found for asteroid {asteroid_number}.")
        return

    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': asteroid_number}))
    if df.empty:
        print(f"No data found in MongoDB for asteroid {asteroid_number}.")
        return

    df["isTrue"] = df["id"].apply(lambda x: x in ztf_predictions)
    df = df.sort_values(by="jd").reset_index(drop=True)

    jd_differences = []
    streak_start_idx = None
    streak_length = 0

    for i, row in df.iterrows():
        if row["isTrue"]:
            if streak_start_idx is None:
                streak_start_idx = i
            streak_length += 1
        else:
            if streak_length >= min_consecutive:
                first = df.loc[streak_start_idx]
                last = df.loc[i - 1]
                jd_differences.append(round(float(last["jd"] - first["jd"]), 2))
            streak_start_idx = None
            streak_length = 0

    if streak_length >= min_consecutive:
        first = df.loc[streak_start_idx]
        last = df.iloc[-1]
        jd_differences.append(round(float(last["jd"] - first["jd"]), 2))

    if not jd_differences:
        print(f"No streaks of {min_consecutive} or more consecutive Trues found for asteroid {asteroid_number}.")
        return

    plot_histogram(jd_differences, bins=np.arange(0, 75, 5),
                   xlabel="JD Difference",
                   ylabel="Number of Streaks",
                   title=f"Histogram of JD Differences for Streaks (Asteroid {asteroid_number})")

# ---------- Main Control Block ----------
if __name__ == "__main__":
    json_file = input("Enter the path to the asteroid JSON file: ").strip()
    group_name = input("Enter the group name for labeling plots (e.g., Group 3): ").strip()

    asteroid_df = analyze_asteroid_group(json_file, group_name)

    user_choice = input("Would you like to analyze a specific asteroid's JD differences? (yes/no): ").strip().lower()
    if user_choice == "yes":
        try:
            asteroid_number = int(input("Enter the asteroid number: ").strip())
            min_consecutive = int(input("Minimum number of consecutive 'True' values to consider a streak: ").strip())
            find_consecutive_true_jd_differences_hist(asteroid_number, min_consecutive, json_file)
        except ValueError:
            print("Invalid asteroid number or minimum consecutive value.")
