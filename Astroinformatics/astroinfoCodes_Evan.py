import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pymongo
from alerce.core import Alerce
from skimage.transform import resize
import torch.nn as nn
import torch.optim as optim
import warnings
from astropy.io.fits.verify import VerifyWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import glob
import os

def dataList(csv_file):
    # Load the entire CSV into a DataFrame
    df = pd.read_csv(csv_file)  # Assumes first row is header by default

    all_ztf_ids = []  # Initialize the list to collect final ZTF IDs

    for ztfid in df.iloc[:, 0]:  # Iterate through the first column (assumed to hold the IDs)
        # Query using the ID to get the associated object info
        data0 = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'id': ztfid}))

        # If there's no match, skip to next
        if data0.empty or "ssnamenr" not in data0:
            continue

        # Get the ssnamenr value and convert it to integer
        astNum = int(data0["ssnamenr"].iloc[0])

        # Use the astNum to get all ZTF entries associated with that asteroid
        data = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': astNum}))

        # Convert the result to a list of IDs and add them to the full list
        ztf = data["id"].tolist()
        all_ztf_ids.extend(ztf)

    return all_ztf_ids


def get_unique_ssids(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Assume SSIDs are in the second column
    ssid_column = df.iloc[:, 1]

    # Drop missing values and get unique values
    unique_ssids = ssid_column.dropna().unique()

    # Convert to a list of integers
    int_ssids = [int(ssid) for ssid in unique_ssids]

    return int_ssids


def find_asteroids_with_10_trues_in_a_row(json_filename):
    # Open the JSON file for reading
    with open(json_filename, 'r') as f:
        data = json.load(f)  # Parse the JSON content into a Python object (usually list or dict)

    # If the JSON structure is a single asteroid dict, wrap it in a list for consistency
    if not isinstance(data, list):
        data = [data]  # Convert single dict to list of one dict

    asteroids_with_10_trues = []  # Initialize a list to hold asteroid numbers with a valid streak

    # Loop over each asteroid in the dataset
    for asteroid in data:
        detections = asteroid.get("detections", [])  # Get the list of detections (default to empty list if missing)
        true_streak = 0  # Initialize the streak counter for consecutive "True" predictions

        # Loop through each detection for the current asteroid
        for detection in detections:
            # Normalize the "predicted" value: make lowercase and remove surrounding whitespace
            predicted = detection.get("predicted", "").strip().lower()

            # If the prediction is "true", increase the streak count
            if predicted == "true":
                true_streak += 1

                # If streak reaches 10, record asteroid number and break out of loop
                if true_streak == 10:
                    asteroids_with_10_trues.append(int(asteroid["AsteroidNum"]))  # Store asteroid number as an integer
                    break  # Stop checking this asteroid—streak already found
            else:
                true_streak = 0  # Reset the streak if prediction is not "true"

    return asteroids_with_10_trues  # Return the list of qualifying asteroid numbers


#####
# Connect to your MongoDB database
pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

# Load the JSON file that contains asteroid detection data
with open("your_file.json", 'r') as f:
    asteroid_data = json.load(f)

# Ensure the JSON data is treated as a list
if not isinstance(asteroid_data, list):
    asteroid_data = [asteroid_data]

# Create a dictionary to map asteroid number to ZTF ID predictions
# Format: {AsteroidNum: {ZTF_ID: True/False}}
predictions_lookup = {}
for asteroid in asteroid_data:
    ast_num = int(asteroid["AsteroidNum"])                        # Convert asteroid number to integer
    if ast_num not in predictions_lookup:
        predictions_lookup[ast_num] = {}                          # Initialize inner dict for this asteroid
    for detection in asteroid.get("detections", []):
        ztf_id = detection.get("id")                              # Get the ZTF id
        pred_value = detection.get("predicted", "").strip().lower()
        if pred_value == "true":
            predictions_lookup[ast_num][ztf_id] = True            # Store True if predicted is "True"
        elif pred_value == "false":
            predictions_lookup[ast_num][ztf_id] = False           # Store False if predicted is "False"

# List to collect each processed asteroid DataFrame
combined_frames = []

# For each asteroid number that has prediction data
for ast_num in predictions_lookup:
    # Load the MongoDB table for this asteroid
    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': ast_num}))

    # Add new column "isTrue" by matching ZTF id to prediction dictionary
    df["isTrue"] = df["id"].map(predictions_lookup[ast_num])     # Returns True/False or NaN if not found

    # Replace NaN with None for clarity
    df["isTrue"] = df["isTrue"].where(pd.notnull(df["isTrue"]), None)

    # Sort the table in ascending order by Julian Date
    df = df.sort_values(by="jd", ascending=True)

    # Add this processed DataFrame to the final list
    combined_frames.append(df)

# Combine all individual asteroid DataFrames into one
final_df = pd.concat(combined_frames, ignore_index=True)

# Drop rows where isTrue is None (NaN)
asteroid_df = asteroid_df.dropna(subset=["isTrue"])

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Specify the asteroid number you want to look up
target_ast_num = 16833

# Filter the DataFrame for this asteroid number
asteroid_df = final_df[final_df["ssnamenr"] == target_ast_num]

# Drop rows where isTrue is None (NaN)
asteroid_df = asteroid_df.dropna(subset=["isTrue"])

# Display the cleaned result
asteroid_df
#####


#####
# Connect to MongoDB
pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

# Load the JSON file that contains asteroid detection data
with open("predictions2.json", 'r') as f:
    asteroid_data = json.load(f)

# Ensure the JSON data is treated as a list
if not isinstance(asteroid_data, list):
    asteroid_data = [asteroid_data]

# ---- USER INPUT: specify target asteroid number ----
target_ast_num = 16833  # Change this to the asteroid number you're interested in

# ---- Find prediction data for the target asteroid ----
# Build a lookup of ZTF ID → True/False for just the selected asteroid
ztf_predictions = {}  # Create an empty dictionary to store ZTF ID → prediction (True/False)

# Loop through each asteroid entry in the loaded JSON data
for asteroid in asteroid_data:
    # Check if the current asteroid number matches the one the user wants
    if int(asteroid.get("AsteroidNum", -1)) == target_ast_num:
        
        # If it matches, loop through all detection records for that asteroid
        for detection in asteroid.get("detections", []):
            ztf_id = detection.get("id")  # Get the ZTF ID string for this detection
            pred_value = detection.get("predicted", "").strip().lower()  # Normalize prediction text (remove spaces and lowercase)
            
            # If the prediction is "true", store True in the dictionary for that ZTF ID
            if pred_value == "true":
                ztf_predictions[ztf_id] = True
            # If the prediction is "false", store False in the dictionary for that ZTF ID
            elif pred_value == "false":
                ztf_predictions[ztf_id] = False
        
        break  # Stop looping since we found the target asteroid and processed its detections
# ---- Query MongoDB for table of this asteroid ----
df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': target_ast_num}))

# Add isTrue column by mapping ZTF ID to predictions
df["isTrue"] = df["id"].map(ztf_predictions)

# Drop rows where prediction is missing (NaN)
df = df.dropna(subset=["isTrue"])

# Sort by Julian date
df = df.sort_values(by="jd", ascending=True)

# Display full DataFrame
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print(df)
#####



#####
# connect to Mongo
pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

# load the json file
with open("predictions2.json", 'r') as f:
    asteroid_data = json.load(f)

# treat the json data as a list
if not isinstance(asteroid_data, list):
    asteroid_data = [asteroid_data]

# ----  specify target asteroid number ----#
target_ast_num = 16833

# Create an empty dictionary to store predictions
ztf_predictions = {}

# loop through each asteroid entry
for asteroid in asteroid_data:
    # check if the current asteroid number matches the target
    if int(asteroid.get("AsteroidNum", -1)) == target_ast_num:
        
        # if it matches, loop through all detections
        for detection in asteroid.get("detections", []):
            # get the ZTF ID string for this detection
            ztf_id = detection.get("id") 
            # remove spaces and lowercase
            pred_value = detection.get("predicted", "").strip().lower() 
            
            # if the prediction is "true", store True
            if pred_value == "true":
                ztf_predictions[ztf_id] = True
            # if the prediction is "false", store False
            elif pred_value == "false":
                ztf_predictions[ztf_id] = False
        # break loop
        break  

# query Mongo for table of the asteroid
df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': target_ast_num}))

# add isTrue column
df["isTrue"] = df["id"].map(ztf_predictions)

# drop rows where prediction is 'None'
df = df.dropna(subset=["isTrue"])

# sort by Julian date
df = df.sort_values(by="jd", ascending=True)

# display full DataFrame
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

df

# Compute the difference between consecutive jd values
jd_diffs = df["jd"].diff().dropna()

# Round all jd differences to 2 significant digits
jd_diffs = jd_diffs.apply(lambda x: float(f"{x:.2g}"))

# Calculate min, max, average, and mode
min_diff = float(f"{jd_diffs.min():.2g}")
max_diff = float(f"{jd_diffs.max():.2g}")
avg_diff = float(f"{jd_diffs.mean():.2g}")

# Use mode(), which can return multiple values — we take the first most common one
mode_diff_series = jd_diffs.mode()
mode_diff = float(f"{mode_diff_series.iloc[0]:.2g}") if not mode_diff_series.empty else None

# Display results
print(f"Minimum difference in jd: {min_diff}")
print(f"Maximum difference in jd: {max_diff}")
print(f"Average difference in jd: {avg_diff}")
print(f"Mode of jd differences: {mode_diff}")
######



def jd_difference(ztf_id_1, ztf_id_2):
    # Connect to Mongo
    pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

    # Load the JSON file
    with open("predictions2.json", 'r') as f:
        asteroid_data = json.load(f)

    # Ensure it's a list
    if not isinstance(asteroid_data, list):
        asteroid_data = [asteroid_data]

    # Find the asteroid containing the first ZTF ID
    target_ast_num = None
    for asteroid in asteroid_data:
        for detection in asteroid.get("detections", []):
            if detection.get("id") == ztf_id_1:
                target_ast_num = int(asteroid.get("AsteroidNum"))
                break
        if target_ast_num is not None:
            break

    if target_ast_num is None:
        return f"ZTF ID '{ztf_id_1}' not found in the predictions JSON."

    # Query Mongo for that asteroid’s full table
    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': target_ast_num}))

    # Ensure both IDs are in the DataFrame
    if ztf_id_1 not in df["id"].values:
        return f"{ztf_id_1} not found in database."
    if ztf_id_2 not in df["id"].values:
        return f"{ztf_id_2} not found in database."

    # Get jd values
    jd1 = df[df["id"] == ztf_id_1]["jd"].values[0]
    jd2 = df[df["id"] == ztf_id_2]["jd"].values[0]

    # Return the positive difference
return round(float(abs(jd1 - jd2)), 2)


def find_consecutive_true_jd_differences(asteroid_number, min_consecutive=15):
    # Connect to Mongo
    pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")
    
    # Load the JSON file
    with open("predictions2.json", 'r') as f:
        asteroid_data = json.load(f)

    # Make sure data is a list
    if not isinstance(asteroid_data, list):
        asteroid_data = [asteroid_data]

    # Extract ZTF IDs with prediction == True for this asteroid
    ztf_predictions = set()
    for asteroid in asteroid_data:
        if int(asteroid.get("AsteroidNum")) == asteroid_number:
            for detection in asteroid.get("detections", []):
                if detection.get("predicted", "").strip().lower() == "true":
                    ztf_predictions.add(detection.get("id"))
            break

    if not ztf_predictions:
        return f"No predicted True values found for asteroid {asteroid_number}."

    # Query MongoDB for this asteroid's data
    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': asteroid_number}))

    if df.empty:
        return f"No data found in MongoDB for asteroid {asteroid_number}."

    # Add 'isTrue' column (keep the whole DataFrame intact)
    df["isTrue"] = df["id"].apply(lambda x: x in ztf_predictions)

    # Sort by Julian Date
    df = df.sort_values(by="jd").reset_index(drop=True)

    # Now scan for streaks of consecutive True values
    results = []
    max_streak = 0
    max_entry = None

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
                jd_diff = round(float(last["jd"] - first["jd"]), 2)
                results.append((streak_length, first["id"], last["id"], jd_diff))
                if streak_length > max_streak:
                    max_streak = streak_length
                    max_entry = (streak_length, first["id"], last["id"], jd_diff)
            # Reset streak
            streak_start_idx = None
            streak_length = 0

    # Handle streak at the very end of the DataFrame
    if streak_length >= min_consecutive:
        first = df.loc[streak_start_idx]
        last = df.iloc[-1]
        jd_diff = round(float(last["jd"] - first["jd"]), 2)
        results.append((streak_length, first["id"], last["id"], jd_diff))
        if streak_length > max_streak:
            max_streak = streak_length
            max_entry = (streak_length, first["id"], last["id"], jd_diff)

    # Output formatting
    if not results:
        return f"No streaks of {min_consecutive} or more consecutive Trues found for asteroid {asteroid_number}."

    output_lines = []
    for streak_length, first_id, last_id, jd_diff in results:
        output_lines.append(
            f"{streak_length} consecutive trues found: Difference in first jd {first_id} and last jd {last_id} is {jd_diff}"
        )

    if max_entry:
        output_lines.append(
            f"Max {max_entry[0]} consecutive trues found: Difference in first jd {max_entry[1]} and last jd {max_entry[2]} is {max_entry[3]}"
        )

    return "\n".join(output_lines)


def find_consecutive_true_jd_differences_hist(asteroid_number, min_consecutive, json_file):
    pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

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
        return f"No predicted True values found for asteroid {asteroid_number}."

    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': asteroid_number}))

    if df.empty:
        return f"No data found in MongoDB for asteroid {asteroid_number}."

    df["isTrue"] = df["id"].apply(lambda x: x in ztf_predictions)
    df = df.sort_values(by="jd").reset_index(drop=True)

    results = []
    jd_differences = []
    max_streak = 0
    max_entry = None

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
                jd_diff = round(float(last["jd"] - first["jd"]), 2)
                results.append((streak_length, first["id"], last["id"], jd_diff))
                jd_differences.append(jd_diff)
                if streak_length > max_streak:
                    max_streak = streak_length
                    max_entry = (streak_length, first["id"], last["id"], jd_diff)
            streak_start_idx = None
            streak_length = 0

    # Handle final streak at end
    if streak_length >= min_consecutive:
        first = df.loc[streak_start_idx]
        last = df.iloc[-1]
        jd_diff = round(float(last["jd"] - first["jd"]), 2)
        results.append((streak_length, first["id"], last["id"], jd_diff))
        jd_differences.append(jd_diff)
        if streak_length > max_streak:
            max_streak = streak_length
            max_entry = (streak_length, first["id"], last["id"], jd_diff)

    if not results:
        return f"No streaks of {min_consecutive} or more consecutive Trues found for asteroid {asteroid_number}."

    bins = np.arange(0, 75, 5)
    n, bins, patches = plt.hist(jd_differences, bins=bins)
    for count, edge_left, patch in zip(n, bins, patches):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y + 0.5, str(int(count)), ha='center', va='bottom', fontsize=9)
    plt.xticks(bins)
    plt.xlim(0, 60)
    plt.ylim(0, 25)
    plt.title(f"Histogram of JD Differences for Streaks (Asteroid {asteroid_number})")
    plt.xlabel("JD Difference")
    plt.ylabel("Number of Streaks")
    plt.show()


#####
# Suppress FITS keyword warnings
warnings.simplefilter('ignore', category=VerifyWarning)

# Asteroid list
asteroid_nums = [
    10026.0, 12331.0, 28279.0, 26986.0, 182678.0, 6232.0, 7409.0, 3171.0, 20857.0, 6052.0,
    13420.0, 13621.0, 29161.0, 11658.0, 12458.0, 1167.0, 86017.0, 44351.0, 462.0, 61088.0,
    12073.0, 1125.0, 1194.0, 34867.0
]

# CNN Module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
try:
    model.load_state_dict(torch.load("cnn_model_updated.pth", map_location=device, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load("cnn_model_updated.pth", map_location=device))
model.eval()

# Connect Clients
client = Alerce()
pmCli = pymongo.MongoClient("mongodb://eclark:vkU0eRXGPuy7L4rkE65w@cmp4818.computers.nau.edu:27017/?authSource=admin")

# Process a single asteroid
def process_asteroid(AsteroidNum):
    db = pmCli["ztf"]["mag18o8"]
    data = pd.DataFrame(db.find({"ssnamenr": AsteroidNum}))

    if "id" not in data.columns:
        return None

    ztf_ids = data["id"].tolist()
    detections = []

    for fits_file in ztf_ids:
        try:
            stamps = client.get_stamps(fits_file)
            difference_image = stamps[2].data
            difference_image = np.nan_to_num(difference_image)
            difference_image = (difference_image - np.min(difference_image)) / (np.max(difference_image) - np.min(difference_image))

            if difference_image.shape != (64, 64):
                difference_image = resize(difference_image, (64, 64), mode="reflect", anti_aliasing=True)

            image_tensor = torch.tensor(difference_image[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                label = "True" if predicted.item() == 1 else "False"

            detections.append({"id": fits_file, "predicted": label})

        except Exception:
            continue

    return {"AsteroidNum": AsteroidNum, "detections": detections}

# Parallel evaluation
def evaluate_candidates_parallel(asteroid_nums, max_workers=4):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_asteroid, num): num for num in asteroid_nums}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔭 Evaluating Asteroids"):
            result = future.result()
            if result:
                results.append(result)
    return results

# Main execution
if __name__ == "__main__":
    results = evaluate_candidates_parallel(asteroid_nums, max_workers=4)

    # Print results
    for asteroid in results:
        print(f"\nResults for Asteroid {asteroid['AsteroidNum']}:")
        for det in asteroid["detections"]:
            print(f"  ID: {det['id']} → Predicted: {det['predicted']}")

    # Save to JSON
    with open("predictions3i.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Saved predictions to predictions3i.json")
#####


#####
# Set your directory containing the 10 JSON files
json_directory = "."  # Change this to your actual directory path

# Pattern to find all .json files
json_files = glob.glob(os.path.join(json_directory, "*.json"))

# Store combined data here
combined_data = []

# Read and append data from each file
for file_path in json_files:
    with open(file_path, "r") as f:
        data = json.load(f)
        combined_data.extend(data)

# Write the combined result to a new JSON file
with open("master_asteroid_predictions.json", "w") as f:
    json.dump(combined_data, f, indent=2)

print(f"✅ Combined {len(json_files)} files into 'master_asteroid_predictions.json' with {len(combined_data)} entries.")
#####


def get_asteroid_dataframe(asteroid_number, json_file="predictions3_master.json"):
    # Connect to MongoDB
    pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

    # Load the JSON file
    with open(json_file, 'r') as f:
        asteroid_data = json.load(f)

    # Ensure the data is a list
    if not isinstance(asteroid_data, list):
        asteroid_data = [asteroid_data]

    # Find the dictionary for the desired asteroid number
    target_ast = None
    for asteroid in asteroid_data:
        if int(asteroid.get("AsteroidNum", -1)) == asteroid_number:
            target_ast = asteroid
            break

    if target_ast is None:
        print(f"❌ Asteroid {asteroid_number} not found in the prediction file.")
        return None

    # Build lookup dictionary for that asteroid's ZTF predictions
    prediction_dict = {}
    for detection in target_ast.get("detections", []):
        ztf_id = detection.get("id")
        predicted_str = detection.get("predicted", "").strip().lower()
        prediction_dict[ztf_id] = predicted_str == "true"

    # Query MongoDB for this asteroid number
    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': asteroid_number}))

    if df.empty:
        print(f"⚠️ No MongoDB records found for asteroid {asteroid_number}.")
        return None

    # Add isTrue column using the prediction dictionary
    df["isTrue"] = df["id"].map(prediction_dict).where(pd.notnull(df["id"]), None)

    # Sort by Julian Date ascending
    df = df.sort_values(by="jd", ascending=True)

    return df