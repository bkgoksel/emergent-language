from typing import NamedTuple, Any, List
import numpy as np
import constants

DEFAULT_NUM_EPOCHS = 5000

DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_FEAT_VEC_SIZE = 256
DEFAULT_TIME_HORIZON = 500

VOCAB_SIZE = 32

DEFAULT_WORLD_DIM = 16
MAX_AGENTS = 3
MAX_LANDMARKS = 3
USE_STRICT_COLORS = True
STRICT_COLORS = np.array([[constants.COLOR_SCALE, 0, 0], [0, constants.COLOR_SCALE, 0], [0, 0, constants.COLOR_SCALE]])
USE_SHAPES = True
NUM_SHAPES = 2

TrainingConfig = NamedTuple('TrainingConfig', [
    ('num_epochs', int)
    ])

default_training_config = TrainingConfig(
        num_epochs=DEFAULT_NUM_EPOCHS)

GameConfig = NamedTuple('GameConfig', [
    ('world_dim', Any),
    ('max_agents', int),
    ('max_landmarks', int),
    ('use_strict_colors', bool),
    ('strict_colors', Any),
    ('use_shapes', bool),
    ('num_shapes', int),
    ('vocab_size', int),
    ('memory_size', int)
])

default_game_config = GameConfig(
        DEFAULT_WORLD_DIM,
        MAX_AGENTS,
        MAX_LANDMARKS,
        USE_STRICT_COLORS,
        STRICT_COLORS,
        USE_SHAPES,
        NUM_SHAPES,
        VOCAB_SIZE,
        DEFAULT_HIDDEN_SIZE)

ProcessingModuleConfig = NamedTuple('ProcessingModuleConfig', [
    ('input_size', int),
    ('hidden_size', int),
    ('dropout', float)
    ])

GoalPredictingProcessingModuleConfig = NamedTuple("GoalPredictingProcessingModuleConfig", [
    ('processor', ProcessingModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('goal_size', int)
    ])

ActionModuleConfig = NamedTuple("ActionModuleConfig", [
    ('goal_processor', ProcessingModuleConfig),
    ('action_processor', ProcessingModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('movement_dim_size', int),
    ('movement_step_size', int),
    ('vocab_size', int)
    ])

AgentModuleConfig = NamedTuple("AgentModuleConfig", [
    ('time_horizon', int),
    ('feat_vec_size', int),
    ('movement_dim_size', int),
    ('goal_size', int),
    ('vocab_size', int),
    ('utterance_processor', GoalPredictingProcessingModuleConfig),
    ('physical_processor', ProcessingModuleConfig),
    ('action_processor', ActionModuleConfig),
    ])

def get_processor_config_with_input_size(input_size):
    return ProcessingModuleConfig(
        input_size=input_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT)

default_goal_predicting_module_config = GoalPredictingProcessingModuleConfig(
    processor=get_processor_config_with_input_size(VOCAB_SIZE),
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    goal_size=constants.GOAL_SIZE)

default_action_module_config = ActionModuleConfig(
        goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
        action_processor=get_processor_config_with_input_size(DEFAULT_FEAT_VEC_SIZE*3),
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        movement_step_size=constants.MOVEMENT_STEP_SIZE,
        vocab_size=VOCAB_SIZE)

default_agent_module_config = AgentModuleConfig(
        time_horizon=DEFAULT_TIME_HORIZON,
        feat_vec_size=DEFAULT_FEAT_VEC_SIZE,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        utterance_processor=default_goal_predicting_module_config,
        physical_processor=get_processor_config_with_input_size(constants.MOVEMENT_DIM_SIZE + constants.PHYSICAL_EMBED_SIZE),
        action_processor=default_action_module_config,
        goal_size=constants.GOAL_SIZE,
        vocab_size=VOCAB_SIZE)

