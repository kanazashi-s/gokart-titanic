import category_encoders as ce
from utils.template import GokartTask

from preprocess.train.drop_target import DropTrainTargetCol
from load_data.inference.load_csv import LoadInferenceCsv


class CategoryEncode(GokartTask):
    category_columns = [
        "Pclass",
        "Sex",
        "Ticket",
        "Cabin",
        "Embarked",
    ]

    def requires(self):
        return dict(
            train_data=DropTrainTargetCol(),
            inference_data=LoadInferenceCsv()
        )

    def output(self):
        return dict(
            train_df=self.make_target("train_df_encoded.csv"),
            inference_df=self.make_target("inference_df_encoded.csv"),
            ce_ordinal=self.make_target("ce_ordinal.pkl")
        )

    def run(self):
        train_df = self.load_data_frame(
            'train_data',
            required_columns=set(self.category_columns) | {"PassengerId", "Name", "Age", "SibSp", "Parch", "Fare"}
        )
        inference_df = self.load('inference_data')

        ce_ordinal = ce.OrdinalEncoder(cols=self.category_columns)
        train_df = ce_ordinal.fit_transform(train_df)
        inference_df = ce_ordinal.transform(inference_df)
        self.dump(train_df, "train_df")
        self.dump(inference_df, "inference_df")
        self.dump(ce_ordinal, "ce_ordinal")
