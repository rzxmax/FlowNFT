"""Shared dependencies and logging for the mechanically extracted trainers."""

from train.common.bootstrap import configure_import_paths

configure_import_paths()

import contextlib
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent import futures
from functools import partial

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
import numpy as np
import torch
import tqdm as tqdm_module
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import (
    ActionDecoder,
    RobotActionPipeline,
    pipeline_with_logprob,
)
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import ode_step_with_logprob
from models.unet import ConditionalUnet1D


tqdm = partial(tqdm_module.tqdm, dynamic_ncols=True)

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = get_logger('flownft.train')
logger.logger.setLevel(logging.INFO)
for handler in logger.logger.handlers[:]:
    logger.logger.removeHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.logger.addHandler(handler)
logger.logger.propagate = False
