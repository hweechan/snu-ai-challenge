import pandas as pd
from torch.utils.data import Dataset
from utils import load_images, parse_answer


class SceneOrderingDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = load_images(self.data_dir, row)
        sentence = row["Sentence"]

        sample = {
            "id": row["Id"],
            "images": images,
            "sentence": sentence,
        }

        if self.is_train:
            sample["answer"] = parse_answer(row["Answer"])
            sample["no_ordering"] = row["No_ordering"]

        return sample
