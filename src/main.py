import gokart
from inference.lgbm import InferLgbmClassifier


if __name__ == '__main__':
    gokart.add_config('conf/param.ini')
    gokart.run()
