from typing import NamedTuple, Any, List
import numpy as np
import constants

DEFAULT_NUM_EPOCHS = 50000
DEFAULT_LR = 1e-4

DEFAULT_HIDDEN_SIZE = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_FEAT_VEC_SIZE = 128
DEFAULT_TIME_HORIZON = 32

USE_UTTERANCES = False
VOCAB_SIZE = 9

DEFAULT_WORLD_DIM = 16
MAX_AGENTS = 3
MAX_LANDMARKS = 3
USE_STRICT_COLORS = True
STRICT_COLORS = np.array([[constants.COLOR_SCALE, 0, 0], [0, constants.COLOR_SCALE, 0], [0, 0, constants.COLOR_SCALE]])
USE_SHAPES = True
NUM_SHAPES = 2

TrainingConfig = NamedTuple('TrainingConfig', [
    ('num_epochs', int),
    ('learning_rate', float)
    ])

default_training_config = TrainingConfig(
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LR)

GameConfig = NamedTuple('GameConfig', [
    ('world_dim', Any),
    ('max_agents', int),
    ('max_landmarks', int),
    ('use_strict_colors', bool),
    ('strict_colors', Any),
    ('use_shapes', bool),
    ('num_shapes', int),
    ('use_utterances', bool),
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
        USE_UTTERANCES,
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
    ('vocab_size', int),
    ('use_utterances', bool)
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
    ('use_utterances', bool),
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

if USE_UTTERANCES:
    feat_size = DEFAULT_FEAT_VEC_SIZE*3
else:
    feat_size = DEFAULT_FEAT_VEC_SIZE*2

default_action_module_config = ActionModuleConfig(
        goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
        action_processor=get_processor_config_with_input_size(feat_size),
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        movement_step_size=constants.MOVEMENT_STEP_SIZE,
        vocab_size=VOCAB_SIZE,
        use_utterances=USE_UTTERANCES)

default_agent_module_config = AgentModuleConfig(
        time_horizon=DEFAULT_TIME_HORIZON,
        feat_vec_size=DEFAULT_FEAT_VEC_SIZE,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        utterance_processor=default_goal_predicting_module_config,
        physical_processor=get_processor_config_with_input_size(constants.MOVEMENT_DIM_SIZE + constants.PHYSICAL_EMBED_SIZE),
        action_processor=default_action_module_config,
        goal_size=constants.GOAL_SIZE,
        vocab_size=VOCAB_SIZE,
        use_utterances=USE_UTTERANCES)

