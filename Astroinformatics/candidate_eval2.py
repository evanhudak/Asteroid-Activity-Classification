import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymongo
from alerce.core import Alerce
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from astropy.io.fits.verify import VerifyWarning

# Suppress FITS keyword warnings
warnings.simplefilter('ignore', category=VerifyWarning)

asteroid_nums = [4446.0,
 112442.0,
 20718.0,
 36881.0,
 2632.0,
 7702.0,
 143486.0,
 2414.0,
 137103.0,
 30465.0,
 1142.0,
 12329.0,
 2031.0,
 302111.0,
 65667.0,
 18678.0,
 31743.0,
 26992.0,
 10808.0,
 13599.0,
 51764.0,
 5591.0,
 1186.0,
 62372.0]

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
    # For older PyTorch versions that don't support weights_only
    model.load_state_dict(torch.load("cnn_model_updated.pth", map_location=device))
model.eval()

# Connect Clients
client = Alerce()
pmCli = pymongo.MongoClient("mongodb://eclark:vkU0eRXGPuy7L4rkE65w@cmp4818.computers.nau.edu:27017/?authSource=admin")

# Display Helper
def show_batch(images, titles, batch_size=25, cols=5):
    rows = batch_size // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            ax.set_title(titles[i], fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()

# Evaluate Candidates
def evaluate_candidates(asteroid_nums):
    results = []

    from tqdm import tqdm

    for AsteroidNum in tqdm(asteroid_nums, desc="🔭 Evaluating Asteroids"):
        db = pmCli["ztf"]["mag18o8"]
        data = pd.DataFrame(db.find({"ssnamenr": AsteroidNum}))

        if "id" not in data.columns:
            #print(f"No 'id' field found for asteroid: {AsteroidNum}")
            continue

        ztf_ids = data["id"].tolist()
        print(f"{AsteroidNum}: {len(ztf_ids)} detections found.")

        detections = []

        for fits_file in tqdm(ztf_ids, desc=f"Asteroid {AsteroidNum}", leave=False):
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

            except Exception as e:
                #print(f" Error with {fits_file}: {e}")
                continue
        results.append({"AsteroidNum": AsteroidNum, "detections": detections})

    return results


if __name__ == "__main__":
    results = evaluate_candidates(asteroid_nums)

    # Print the results
    for asteroid in results:
        print(f"\n Results for Asteroid {asteroid['AsteroidNum']}:")
        for det in asteroid["detections"]:
            print(f"  ID: {det['id']} → Predicted: {det['predicted']}")
    import json
    with open("predictions3c.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved predictions to predictions.json")
