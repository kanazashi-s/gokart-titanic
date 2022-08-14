import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from utils.template import GokartTask
from preprocess.common.encode import CategoryEncode
from preprocess.train.get_target import GetTrainTargetCol


class TrainLgbmClassifier(GokartTask):
    use_columns = [
        "Pclass",
        "Sex",
        "Ticket",
        "Cabin",
        "Embarked",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    def requires(self):
        return dict(
            data=CategoryEncode(),
            train_target=GetTrainTargetCol(),
        )

    def output(self):
        return dict(
            lgbm_models=self.make_target("lgbm_model.pkl"),
            require_columns=self.make_target("require_columns.pkl")
        )

    def run(self):
        self.dump(self.use_columns, "require_columns")

        train_df = self.load('data')["train_df"][self.use_columns]
        train_target = self.load('train_target')

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        models = []
        for i, (train_idx, valid_idx) in enumerate(cv.split(train_df)):
            train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]
            valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]

            lgbm_model = lgb.LGBMClassifier()
            lgbm_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=False)
            valid_pred = lgbm_model.predict_proba(valid_x)[:, 1]
            valid_loss = log_loss(valid_y, valid_pred)
            valid_acc = accuracy_score(valid_y, valid_pred > 0.5)
            print(f"fold {i} valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f}")
            models.append(lgbm_model)

        self.dump(models, "lgbm_models")
