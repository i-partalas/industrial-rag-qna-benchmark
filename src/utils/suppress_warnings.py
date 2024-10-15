import os
import warnings

# Suppress TensorFlow info and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress DeepEval UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="You are using deepeval version",
    module="deepeval",
)
