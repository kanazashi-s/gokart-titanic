from pathlib import Path
import pandas as pd
from utils.template import GokartTask


class LoadTrainCsv(GokartTask):
    csv_path = Path("data", "raw", "train.csv")

    def run(self):
        train_df = pd.read_csv(self.csv_path)
        self.dump(train_df)
