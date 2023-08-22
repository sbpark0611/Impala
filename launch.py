import argparse
import os

import numpy as np

import ray
from ray import air, tune
from ray.rllib.algorithms import impala
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import registry

if __name__ == "__main__":
    print("hello world")
