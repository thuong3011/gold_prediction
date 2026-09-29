"""Run data collection -> cleaning -> feature engineering -> XGBoost training."""

from data_pipeline.gold_collector import main as collect_main
from data_pipeline.cleaner import main as clean_main
from data_pipeline.feature_engineering import main as feature_main
from ml.train_xgboost import train as train_xgb


if __name__ == "__main__":
    collect_main()
    clean_main()
    feature_main()
    train_xgb()
