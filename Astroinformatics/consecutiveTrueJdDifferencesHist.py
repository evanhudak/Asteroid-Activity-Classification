def find_consecutive_true_jd_differences_hist(asteroid_number, min_consecutive, json_file):
    # Connect to the MongoDB database
    pmCli = pymongo.MongoClient("mongodb://group6:password@cmp4818.computers.nau.edu:27018")

    # Load the JSON file containing asteroid prediction data
    with open(json_file, 'r') as f:
        asteroid_data = json.load(f)

    # Ensure the data is a list for consistent processing
    if not isinstance(asteroid_data, list):
        asteroid_data = [asteroid_data]

    # Build a set of detection IDs where prediction == "true" for the given asteroid
    ztf_predictions = set()
    for asteroid in asteroid_data:
        if int(asteroid.get("AsteroidNum")) == asteroid_number:
            for detection in asteroid.get("detections", []):
                if detection.get("predicted", "").strip().lower() == "true":
                    ztf_predictions.add(detection.get("id"))
            break  # Exit after finding the target asteroid

    # Exit early if there are no "true" predictions for this asteroid
    if not ztf_predictions:
        return f"No predicted True values found for asteroid {asteroid_number}."

    # Query the MongoDB collection for observational data of the target asteroid
    df = pd.DataFrame(pmCli["ztf"]["mag18o8"].find({'ssnamenr': asteroid_number}))

    # Exit early if no records were found in the database
    if df.empty:
        return f"No data found in MongoDB for asteroid {asteroid_number}."

    # Create a new column 'isTrue' indicating whether each detection ID was predicted true
    df["isTrue"] = df["id"].apply(lambda x: x in ztf_predictions)

    # Sort the DataFrame by Julian Date (jd) to allow streak detection
    df = df.sort_values(by="jd").reset_index(drop=True)

    # Initialize containers for storing results
    results = []         # Stores (length, first ID, last ID, jd difference)
    jd_differences = []  # Stores jd differences of streaks for histogram
    max_streak = 0       # Tracks the longest streak
    max_entry = None     # Stores the details of the longest streak

    # Initialize streak tracking variables
    streak_start_idx = None
    streak_length = 0

    # Iterate through the DataFrame to find consecutive True prediction streaks
    for i, row in df.iterrows():
        if row["isTrue"]:
            if streak_start_idx is None:
                streak_start_idx = i  # Mark the start of a new streak
            streak_length += 1        # Increase streak length
        else:
            # If the current streak meets the minimum required length, save it
            if streak_length >= min_consecutive:
                first = df.loc[streak_start_idx]
                last = df.loc[i - 1]
                jd_diff = round(float(last["jd"] - first["jd"]), 2)
                results.append((streak_length, first["id"], last["id"], jd_diff))
                jd_differences.append(jd_diff)
                # Track the longest streak
                if streak_length > max_streak:
                    max_streak = streak_length
                    max_entry = (streak_length, first["id"], last["id"], jd_diff)
            # Reset streak tracking
            streak_start_idx = None
            streak_length = 0

    # Handle any final streak that may occur at the end of the DataFrame
    if streak_length >= min_consecutive:
        first = df.loc[streak_start_idx]
        last = df.iloc[-1]
        jd_diff = round(float(last["jd"] - first["jd"]), 2)
        results.append((streak_length, first["id"], last["id"], jd_diff))
        jd_differences.append(jd_diff)
        if streak_length > max_streak:
            max_streak = streak_length
            max_entry = (streak_length, first["id"], last["id"], jd_diff)

    # Exit early if no qualifying streaks were found
    if not results:
        return f"No streaks of {min_consecutive} or more consecutive Trues found for asteroid {asteroid_number}."

    # Plot a histogram of the JD differences for all qualifying streaks
    bins = np.arange(0, 75, 5)  # Set histogram bin edges
    n, bins, patches = plt.hist(jd_differences, bins=bins)
    
    # Annotate bars with count values
    for count, edge_left, patch in zip(n, bins, patches):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y + 0.5, str(int(count)), ha='center', va='bottom', fontsize=9)
    
    # Configure plot appearance
    plt.xticks(bins)
    plt.xlim(0, 60)
    plt.ylim(0, 25)
    plt.title(f"Histogram of JD Differences for Streaks (Asteroid {asteroid_number})")
    plt.xlabel("JD Difference")
    plt.ylabel("Number of Streaks")
    plt.show()
