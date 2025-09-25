"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'
GLOBAL_SEED = 42

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_MULTI_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

LIBERO1_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


LIBERO2_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 2,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


LIBERO4_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 4,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

LIBERO16_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

LIBERO24_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 24,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

LIBERO32_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 32,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}


ALOHA10_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 10,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}


ALOHA16_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}


ALOHA50_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 50,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

BRIDGE4_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 4,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


METAWORLD_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 4,
    "PROPRIO_DIM": 4,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}



BRIDGE8_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

RT1_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()
    
    if "multi_li" in cmd_args:
        return "MULTI_LI"
    elif "1li" in cmd_args:
        return "1LI"
    elif "2li" in cmd_args:
        return "2LI"
    elif "4li" in cmd_args:
        return "4LI"
    elif "16_li" in cmd_args:
        return "16LI"
    elif "24_li" in cmd_args:
        return "24LI"
    elif "32_li" in cmd_args:
        return "32LI"

    elif "libero" in cmd_args:
        return "LIBERO"
    elif "10_al" in cmd_args:
        return "ALOHA10"
    elif "16_al" in cmd_args:
        return "ALOHA16"
    elif "50_al" in cmd_args:
        return "ALOHA50"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "4_br" in cmd_args:
        return "4BRI"
    elif "8_br" in cmd_args:
        return "8BRI"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "rt1" in cmd_args:
        return "RT1"
    elif "metaworld" in cmd_args:
        return "METAWORLD"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "MULTI_LI":
    constants = LIBERO_MULTI_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA10":
    constants = ALOHA10_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA16":
    constants = ALOHA16_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA50":
    constants = ALOHA50_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "1LI":
    constants = LIBERO1_CONSTANTS
elif ROBOT_PLATFORM == "2LI":
    constants = LIBERO2_CONSTANTS
elif ROBOT_PLATFORM == "4LI":
    constants = LIBERO4_CONSTANTS
elif ROBOT_PLATFORM == "16LI":
    constants = LIBERO16_CONSTANTS
elif ROBOT_PLATFORM == "24LI":
    constants = LIBERO24_CONSTANTS
elif ROBOT_PLATFORM == "32LI":
    constants = LIBERO32_CONSTANTS
elif ROBOT_PLATFORM == "RT1":
    constants = RT1_CONSTANTS
elif ROBOT_PLATFORM == "4BRI":
    constants = BRIDGE4_CONSTANTS
elif ROBOT_PLATFORM == "8BRI":
    constants = BRIDGE8_CONSTANTS
elif ROBOT_PLATFORM == "METAWORLD":
    constants = METAWORLD_CONSTANTS
else:
    raise ValueError(f"Unsupported robot platform: {ROBOT_PLATFORM}")


# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]

ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
