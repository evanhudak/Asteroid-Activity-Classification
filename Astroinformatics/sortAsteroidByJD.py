import pymongo
import json
import pandas as pd

def get_ztf_predictions(json_file_path, target_ast_num):
    # Connect to MongoDB
    pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

    # Load the JSON file
    with open(json_file_path, 'r') as f:
        asteroid_data = json.load(f)

    # Ensure the JSON data is treated as a list
    if not isinstance(asteroid_data, list):
        asteroid_data = [asteroid_data]

    # ---- Find prediction data for the target asteroid ----
    ztf_predictions = {}  # Dictionary: ZTF ID → prediction (True/False)

    for asteroid in asteroid_data:
        if int(asteroid.get("AsteroidNum", -1)) == target_ast_num:
            for detection in asteroid.get("detections", []):
                ztf_id = detection.get("id")
                pred_value = detection.get("predicted", "").strip().lower()
                if pred_value == "true":
                    ztf_predictions[ztf_id] = True
                elif pred_value == "false":
                    ztf_predictions[ztf_id] = False
            break  # Found and processed the target asteroid

    # ---- Query MongoDB for this asteroid ----
    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': target_ast_num}))

    # Map ZTF ID to predictions and drop rows with missing predictions
    df["isTrue"] = df["id"].map(ztf_predictions)
    df = df.dropna(subset=["isTrue"])

    # Sort by Julian date
    df = df.sort_values(by="jd", ascending=True)

    return df