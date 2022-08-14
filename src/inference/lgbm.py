import numpy as np
import pandas as pd
from utils.template import GokartTask
from preprocess.common.encode import CategoryEncode
from training.lgbm import TrainLgbmClassifier


class InferLgbmClassifier(GokartTask):
    def requires(self):
        return dict(
            data=CategoryEncode(),
            lgbm_models=TrainLgbmClassifier(),
        )

    def output(self):
        return dict(
            submission_csv=self.make_target("submission.csv"),
        )

    def run(self):
        use_columns = self.load('lgbm_models')["require_columns"]
        inference_df = self.load("data")["inference_df"]
        lgbm_models = self.load("lgbm_models")["lgbm_models"]

        pred = []
        for model in lgbm_models:
            pred.append(model.predict_proba(inference_df[use_columns])[:, 1])
        pred = (np.mean(pred, axis=0) > 0.5).astype(int)

        submission_df = pd.DataFrame({"PassengerId": inference_df["PassengerId"], "Survived": pred})
        self.dump(submission_df, "submission_csv")
