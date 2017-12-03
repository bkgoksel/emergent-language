import torch
import torch.nn as nn
import numpy as np
from typing import NamedTuple, Any
from torch import Variable
from game import GameGenerator
import constants



def gumbel_softmax(x):
    y = x + self.sample_gumbel(x.size)
    return nn.softmax(y/self.temperature)

def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)

class ProcessingModule(nn.Module):
    def __init__(self, config):
        super(ProcessingModule, self).__init__()
        self.dropout = nn.dropout(config.dropout)
        self.lstm_cell = nn.LSTMCell(config.input_size, config.hidden_size)
        self.h = torch.Tensor(config.batch_size, config.hidden_size).zero_()
        self.fully_connected = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU())

    def forward(self, x, m):
        self.h, m = self.cell(x, (self.h, m))
        self.h = self.dropout(self.h)
        return self.fully_connected(self.h)

class GoalPredictingProcessingModule(nn.Module):
    def __init__(self, config):
        super(GoalPredictingProcessingModule, self).__init__()
        self.processor = ProcessingModule(config)
        self.goal_predictor = nn.Sequential(
                nn.Linear(config.input_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.goal_size))

    def forward(self, x, mem):
        processed, mem = self.processor(x, mem)
        goal_prediction = self.goal_predictor(processed)
        return processed, mem, goal_prediction

class ActionModule(nn.Module):
    def __init__(self, config):
        super(ActionModule, self).__init__()
        self.goal_processor = ProcessingModule(config)
        self.processor = ProcessingModule(config)
        self.movement_chooser = nn.Sequential(
                nn.Linear(config.input_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.movement_size),
                nn.Softmax()
                )
        self.utterance_chooser = nn.Sequential(
                nn.Linear(config.input_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.vocab_size)
                )

    def forward(self, physical, utterance, goal, mem):
        goal_processed, _ = self.goal_processor(goal, mem)
        x = torch.cat([physical, utterance, goal_processed])
        processed, mem = self.processor(x, mem)
        movement = self.movement_chooser(processed)
        utterance = self.utterance_chooser(processed)
        final_utterance = gumbel_softmax(utterance)
        final_movement = torch.add(torch.mul(movement, 2*self.movement_step_size), -self.movement_step_size)
        return final_movement, final_utterance, mem

MemoryCollection = NamedTuple('MemoryCollection', [
    ('utterance', Any),
    ('physical', Any),
    ('action', Any),
])

class VectorizedGame():
    def __init__(self, agents, landmarks, goals, vocab_size, memory_size) -> None:
        self.num_agents = agents.shape[0]
        self.physical = torch.Tensor(np.concatenate((agents, landmarks)))
        self.goals = torch.Tensor(goals)
        self.utterances = torch.Tensor((self.num_agents, vocab_size)).zero_()
        self.memories = MemoryCollection(
                utterance=torch.Tensor((self.num_agents, self.num_agents, memory_size)).zero_(),
                physical=torch.Tensor((self.num_agents, self.physical.shape[0], memory_size)).zero_(),
                action=torch.Tensor((self.num_agents, memory_size)).zero_())
        agent_baselines = torch.Tensor((self.num_agents, 1, constants.LANDMARK_EMBED_DIM)).zero_()
        agent_view = self.physical.view(self.physical.size()[0], 1, self.physical.size[1])
        agent_baselines[:, 0, 0:2] = self.physical[:self.num_agents, 0:2]
        self.observations = self.physical - agent_baselines

class AgentModule(nn.Module):
    def __init__(self, config):
        super(AgentModule, self).__init__()
        self.batch_size = config.batch_size
        self.time_horizon = config.time_horizon
        self.utterance_processor = ProcessingModule(config.processing)
        self.utterance_pooling = nn.AdaptiveAvgPool1d(config.feat_vec_size)
        self.physical_processor = ProcessingModule(config.processing)
        self.physical_pooling = nn.AdaptiveAvgPool1d(config.feat_vec_size)
        self.action_processor = ActionModule(config.action)
        self.total_reward = Variable((1))
        self.reward_transitor = RewardTransitionModule(config)

    def forward(self):
        # 1- Initialize games:
        pass


    """
        Game:
            -num_agents
            -physical: [num_agents + num_landmarks, entity_embed_size]
            -utterances: [num_agents, vocab_size]
            -goals: [num_agents, goal_size]
            -observations: [num_agents, num_agents + num_landmarks, entity_embed_size]
            -memories
                -utterance: [num_agents, num_agents, memory_size]
                -physical:[num_agents, num_agents + num_landmarks, memory_size]
                -action: [num_agents, memory_size]
    """

    def forward_game(self, game):
        for t in range(self.time_horizon):
            for agent in range(game.num_agents):
                utterance_processes = []
                physical_processes = []
                goal_predictions = {}
                for other_agent in range(game.num_agents):
                    utterance_processed, game.memories.utterance[agent, other_agent], goal_predicted = self.utterance_processor(game.utterances[other_agent], game.memories.utterance[agent, other_agent])
                    utterance_processes.append(utterance_processed)
                    goal_predictions[agent][other_agent] = goal_predicted

                    physical_processed, game.memories.physical[agent, other_agent] = self.physical_processor(game.physical[other_agent], game.memories.physical[agent, other_agent])
                    physical_processes.append(physical_processed)
                for landmark in range(game.num_agents, game.num_agents + game.num_landmarks):
                    physical_processed, game.memories.physical[agent, landmark] = self.physical_processor(game.physical[landmark], game.memories.physical[agent, landmark])
                    physical_processes.append(physical_processed)

                all_utterances = torch.cat(utterance_processes, 0)
                all_physical = torch.cat(physical_processes, 0)

                utterance_feat = self.utterance_pooling(all_utterance)
                physical_feat = self.physical_pooling(all_physical)

                movement, utterance, game.memories.action[agent] = self.action_processor(utterance_feat, physical_feat, game.goals[agent], game.memories.action[agent])
                reward = self.reward_transitor(game, agent, movement, utterance)
                self.total_reward += reward

