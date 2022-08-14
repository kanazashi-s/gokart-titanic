from utils.template import GokartTask
from load_data.train.load_csv import LoadTrainCsv


class GetTrainTargetCol(GokartTask):
    def requires(self):
        return LoadTrainCsv()

    def run(self):
        train_df = self.load()
        train_y = train_df['Survived']
        self.dump(train_y)
