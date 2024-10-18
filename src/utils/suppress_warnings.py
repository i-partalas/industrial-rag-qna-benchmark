import logging
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

# Silence logger warnings from the Unstructured module
logging.getLogger("unstructured").setLevel(logging.ERROR)

# Silence logger warnings from the PDFMiner module
logging.getLogger("pdfminer").setLevel(logging.ERROR)
