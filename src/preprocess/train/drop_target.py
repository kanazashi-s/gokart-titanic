from utils.template import GokartTask
from load_data.train.load_csv import LoadTrainCsv


class DropTrainTargetCol(GokartTask):
    def requires(self):
        return LoadTrainCsv()

    def run(self):
        train_df = self.load()
        train_df = train_df.drop(columns=['Survived'])
        self.dump(train_df)
