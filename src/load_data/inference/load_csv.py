from pathlib import Path
import pandas as pd
from utils.template import GokartTask


class LoadInferenceCsv(GokartTask):
    csv_path = Path("data", "raw", "test.csv")

    def run(self):
        test_df = pd.read_csv(self.csv_path)
        self.dump(test_df)
