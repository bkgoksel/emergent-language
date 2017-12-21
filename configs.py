import pdb
from typing import NamedTuple, Any, List
import numpy as np
import constants

DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LR = 5e-5
SAVE_MODEL = True
DEFAULT_MODEL_FILE = 'latest.pt'

DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_FEAT_VEC_SIZE = 256
DEFAULT_TIME_HORIZON = 16

USE_UTTERANCES = True
PENALIZE_WORDS = True
DEFAULT_VOCAB_SIZE = 20
DEFAULT_OOV_PROB = 5

DEFAULT_WORLD_DIM = 16
MAX_AGENTS = 3
MAX_LANDMARKS = 3
MIN_AGENTS = 2
MIN_LANDMARKS = 3
NUM_COLORS = 3
NUM_SHAPES = 2

TrainingConfig = NamedTuple('TrainingConfig', [
    ('num_epochs', int),
    ('learning_rate', float),
    ('load_model', bool),
    ('load_model_file', str),
    ('save_model', bool),
    ('save_model_file', str),
    ('use_cuda', bool)
    ])

GameConfig = NamedTuple('GameConfig', [
    ('batch_size', int),
    ('world_dim', Any),
    ('max_agents', int),
    ('max_landmarks', int),
    ('min_agents', int),
    ('min_landmarks', int),
    ('num_shapes', int),
    ('num_colors', int),
    ('use_utterances', bool),
    ('vocab_size', int),
    ('memory_size', int),
    ('use_cuda', bool),
])

ProcessingModuleConfig = NamedTuple('ProcessingModuleConfig', [
    ('input_size', int),
    ('hidden_size', int),
    ('dropout', float)
    ])

WordCountingModuleConfig = NamedTuple('WordCountingModuleConfig', [
    ('vocab_size', int),
    ('oov_prob', float),
    ('use_cuda', bool)
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
    ('use_utterances', bool),
    ('use_cuda', bool)
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
    ('word_counter', WordCountingModuleConfig),
    ('use_utterances', bool),
    ('penalize_words', bool),
    ('use_cuda', bool)
    ])

default_training_config = TrainingConfig(
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LR,
        load_model=False,
        load_model_file="",
        save_model=SAVE_MODEL,
        save_model_file=DEFAULT_MODEL_FILE,
        use_cuda=False)

default_word_counter_config = WordCountingModuleConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        oov_prob=DEFAULT_OOV_PROB,
        use_cuda=False)

default_game_config = GameConfig(
        DEFAULT_BATCH_SIZE,
        DEFAULT_WORLD_DIM,
        MAX_AGENTS,
        MAX_LANDMARKS,
        MIN_AGENTS,
        MIN_LANDMARKS,
        NUM_SHAPES,
        NUM_COLORS,
        USE_UTTERANCES,
        DEFAULT_VOCAB_SIZE,
        DEFAULT_HIDDEN_SIZE,
        False
        )

if USE_UTTERANCES:
    feat_size = DEFAULT_FEAT_VEC_SIZE*3
else:
    feat_size = DEFAULT_FEAT_VEC_SIZE*2

def get_processor_config_with_input_size(input_size):
    return ProcessingModuleConfig(
        input_size=input_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT)

default_action_module_config = ActionModuleConfig(
        goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
        action_processor=get_processor_config_with_input_size(feat_size),
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        movement_step_size=constants.MOVEMENT_STEP_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        use_utterances=USE_UTTERANCES,
        use_cuda=False)

default_goal_predicting_module_config = GoalPredictingProcessingModuleConfig(
    processor=get_processor_config_with_input_size(DEFAULT_VOCAB_SIZE),
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    goal_size=constants.GOAL_SIZE)

default_agent_config = AgentModuleConfig(
        time_horizon=DEFAULT_TIME_HORIZON,
        feat_vec_size=DEFAULT_FEAT_VEC_SIZE,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        utterance_processor=default_goal_predicting_module_config,
        physical_processor=get_processor_config_with_input_size(constants.MOVEMENT_DIM_SIZE + constants.PHYSICAL_EMBED_SIZE),
        action_processor=default_action_module_config,
        word_counter=default_word_counter_config,
        goal_size=constants.GOAL_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        use_utterances=USE_UTTERANCES,
        penalize_words=PENALIZE_WORDS,
        use_cuda=False)

def get_training_config(kwargs):
    return TrainingConfig(
            num_epochs=kwargs['n_epochs'] or default_training_config.num_epochs,
            learning_rate=kwargs['learning_rate'] or default_training_config.learning_rate,
            load_model=bool(kwargs['load_model_weights']),
            load_model_file=kwargs['load_model_weights'] or default_training_config.load_model_file,
            save_model=default_training_config.save_model,
            save_model_file=kwargs['save_model_weights'] or default_training_config.save_model_file,
            use_cuda=kwargs['use_cuda'])

def get_game_config(kwargs):
    return GameConfig(
            batch_size=kwargs['batch_size'] or default_game_config.batch_size,
            world_dim=kwargs['world_dim'] or default_game_config.world_dim,
            max_agents=kwargs['max_agents'] or default_game_config.max_agents,
            min_agents=kwargs['min_agents'] or default_game_config.min_agents,
            max_landmarks=kwargs['max_landmarks'] or default_game_config.max_landmarks,
            min_landmarks=kwargs['min_landmarks'] or default_game_config.min_landmarks,
            num_shapes=kwargs['num_shapes'] or default_game_config.num_shapes,
            num_colors=kwargs['num_colors'] or default_game_config.num_colors,
            use_utterances=not kwargs['no_utterances'],
            vocab_size=kwargs['vocab_size'] or default_game_config.vocab_size,
            memory_size=default_game_config.memory_size,
            use_cuda=kwargs['use_cuda']
            )

def get_agent_config(kwargs):
    vocab_size = kwargs['vocab_size'] or DEFAULT_VOCAB_SIZE
    use_utterances = (not kwargs['no_utterances'])
    use_cuda = kwargs['use_cuda']
    penalize_words = kwargs['penalize_words']
    oov_prob = kwargs['oov_prob'] or DEFAULT_OOV_PROB
    if use_utterances:
        feat_vec_size = DEFAULT_FEAT_VEC_SIZE*3
    else:
        feat_vec_size = DEFAULT_FEAT_VEC_SIZE*2
    utterance_processor = GoalPredictingProcessingModuleConfig(
            processor=get_processor_config_with_input_size(vocab_size),
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            goal_size=constants.GOAL_SIZE)
    action_processor = ActionModuleConfig(
            goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
            action_processor=get_processor_config_with_input_size(feat_vec_size),
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            movement_dim_size=constants.MOVEMENT_DIM_SIZE,
            movement_step_size=constants.MOVEMENT_STEP_SIZE,
            vocab_size=vocab_size,
            use_utterances=use_utterances,
            use_cuda=use_cuda)
    word_counter = WordCountingModuleConfig(
            vocab_size=vocab_size,
            oov_prob=oov_prob,
            use_cuda=use_cuda)

    return AgentModuleConfig(
            time_horizon=kwargs['n_timesteps'] or default_agent_config.time_horizon,
            feat_vec_size=default_agent_config.feat_vec_size,
            movement_dim_size=default_agent_config.movement_dim_size,
            utterance_processor=utterance_processor,
            physical_processor=default_agent_config.physical_processor,
            action_processor=action_processor,
            word_counter=word_counter,
            goal_size=default_agent_config.goal_size,
            vocab_size=vocab_size,
            use_utterances=use_utterances,
            penalize_words=penalize_words,
            use_cuda=use_cuda
            )

