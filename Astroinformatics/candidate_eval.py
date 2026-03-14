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

asteroid_nums = [773.0,
 2090.0,
 70045.0,
 3704.0,
 34746.0,
 6122.0,
 16787.0,
 31733.0,
 16713.0,
 7353.0,
 82207.0,
 6979.0,
 4838.0,
 7093.0,
 29814.0,
 15811.0,
 6129.0,
 2393.0,
 25272.0,
 677.0,
 30928.0,
 1963.0,
 5268.0,
 16438.0,
 5325.0,
 26342.0,
 45663.0,
 5438.0,
 5676.0,
 14835.0,
 49644.0,
 2098.0,
 6021.0,
 11524.0,
 15484.0,
 27994.0,
 5688.0,
 6185.0,
 18942.0,
 53879.0,
 29578.0,
 16466.0,
 4566.0,
 16833.0,
 5067.0,
 50960.0,
 22825.0,
 51304.0,
 48430.0,
 48409.0,
 14607.0,
 20800.0,
 5568.0,
 2576.0,
 476.0,
 37730.0,
 6924.0,
 42571.0,
 29218.0,
 7522.0,
 11321.0,
 48433.0,
 51333.0,
 5087.0,
 57889.0,
 49439.0,
 15983.0,
 14962.0,
 8284.0,
 6477.0,
 332.0,
 10648.0,
 14204.0,
 83604.0,
 12005.0,
 4826.0,
 10713.0,
 9573.0,
 7315.0,
 58192.0,
 28455.0,
 44060.0,
 2257.0,
 627.0,
 55039.0,
 93.0,
 2113.0,
 16352.0,
 1079.0,
 7577.0,
 6730.0,
 32221.0,
 3306.0,
 9414.0,
 110456.0,
 56116.0,
 57341.0,
 6974.0,
 1765.0,
 8168.0,
 13361.0,
 28191.0,
 4088.0,
 19159.0,
 3631.0,
 6401.0,
 89247.0,
 2858.0,
 64215.0,
 30346.0,
 15400.0,
 35076.0,
 413133.0,
 31650.0,
 1033.0,
 316720.0,
 50730.0,
 9003.0,
 101167.0,
 15159.0,
 47493.0,
 19740.0,
 1879.0,
 109275.0,
 20502.0,
 14759.0,
 10376.0,
 66103.0,
 13698.0,
 894.0,
 48540.0,
 15165.0,
 562.0,
 1531.0,
 11987.0,
 3165.0,
 32438.0,
 109582.0,
 26249.0,
 84528.0,
 54523.0,
 11320.0,
 37113.0,
 2767.0,
 143300.0,
 50379.0,
 959.0,
 32041.0,
 34515.0,
 16394.0,
 1395.0,
 1349.0,
 30521.0,
 76094.0,
 55947.0,
 70354.0,
 11202.0,
 17021.0,
 43755.0]

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
    with open("predictions2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved predictions to predictions.json")
