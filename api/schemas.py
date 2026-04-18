from dataclasses import dataclass


@dataclass
class PredictionRequest:
    feature_1: float
    feature_2: float
