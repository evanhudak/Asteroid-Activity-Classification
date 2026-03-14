import pymongo
import pandas as pd
import json

def find_consecutive_true_jd_differences(asteroid_number, min_consecutive, json_file):
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
                    
            streak_start_idx = None
            streak_length = 0

    if streak_length >= min_consecutive:
        first = df.loc[streak_start_idx]
        last = df.iloc[-1]
        jd_diff = round(float(last["jd"] - first["jd"]), 2)
        results.append((streak_length, first["id"], last["id"], jd_diff))
        if streak_length > max_streak:
            max_streak = streak_length
            max_entry = (streak_length, first["id"], last["id"], jd_diff)

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

##----run code with this command --
result = find_consecutive_true_jd_differences(16833, 5, "predictions2.json")
print(result)