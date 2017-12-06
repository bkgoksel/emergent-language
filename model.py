import torch
import torch.nn as nn
import numpy as np
from typing import NamedTuple, Any
from torch import Variable, Tensor
from game import GameGenerator
import constants


# TODO: Configs

def gumbel_softmax(x):
    y = x + self.sample_gumbel(x.size())
    return nn.softmax(y/self.temperature)

def sample_gumbel(shape, eps=1e-20):
    U = Tensor(shape).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)

"""
    A Processing module takes an input from a stream and the independent memory
    of that stream and runs a single timestep of a GRU cell, followed by
    dropout and finally a linear ReLU layer on top of the GRU output.
    It returns the output of the fully connected layer as well as the update to
    the independent memory.
"""
class ProcessingModule(nn.Module):
    def __init__(self, config):
        super(ProcessingModule, self).__init__()
        self.cell = nn.GRUCell(config.input_size, config.hidden_size)
        self.fully_connected = nn.Sequential(
                nn.dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU())

    def forward(self, x, m):
        m = self.cell(x, m)
        return self.fully_connected(m), m

"""
    A GoalPredictingProcessingModule acts like a regular processing module but
    also runs a goal predictor layer that is a two layer fully-connected
    network. It returns the regular processing module's output, its memory
    update and finally a goal vector sized goal prediction
"""
class GoalPredictingProcessingModule(nn.Module):
    def __init__(self, config):
        super(GoalPredictingProcessingModule, self).__init__()
        self.processor = ProcessingModule(config)
        self.goal_predictor = nn.Sequential(
                nn.Linear(config.input_size, config.hidden_size),
                nn.dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.goal_size))

    def forward(self, x, mem):
        processed, mem = self.processor(x, mem)
        goal_prediction = self.goal_predictor(processed)
        return processed, mem, goal_prediction

"""
    An ActionModule takes in the physical observation feature vector, the
    utterance observation feature vector and the individual goal of an agent
    (alongside the memory for the module), processes the goal to turn it into
    a goal feature vector, and runs the concatenation of all three feature
    vectors through a processing module. The output of the processing module
    is then fed into two independent fully connected networks to output
    utterance and movement actions
"""
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

"""
    The GameModule takes in all actions(movement, utterance, goal prediction)
    of all agents for a given timestep and returns the total reward for that
    timestep.

    Game consists of:
        -num_agents (scalar)
        -num_landmarks (scalar)
        -locations: [num_agents + num_landmarks, 2]
        -physical: [num_agents + num_landmarks, entity_embed_size]
        -utterances: [num_agents, vocab_size]
        -goals: [num_agents, goal_size]
        -location_observations: [num_agents, num_agents + num_landmarks, 2]
        -memories
            -utterance: [num_agents, num_agents, memory_size]
            -physical:[num_agents, num_agents + num_landmarks, memory_size]
            -action: [num_agents, memory_size]
"""

class GameModule(nn.Module):
    def __init__(self, agent_locations, agent_physical, landmark_locations, goals, vocab_size, memory_size):
        super(GameModule, self).__init__()
        self.num_agents = agent_locations.shape[0]
        self.num_landmarks = landmark_locations.shape[0]
        self.num_entities = self.num_agents + self.num_entities

        self.locations = torch.from_numpy(np.concatenate((agent_locations, landmark_locations)))
        self.physical = torch.from_numpy(np.concatenate((agent_physical, landmark_physical)))

        self.goals = torch.from_numpy(goals)
        self.utterances = torch.zeros(self.num_agents, vocab_size)
        self.memories = MemoryCollection(
                utterance=torch.zeros(self.num_agents, self.num_agents, memory_size),
                physical=torch.zeros(self.num_agents, self.locations.shape[0], memory_size),
                action=torch.zeros(self.num_agents, memory_size))

        agent_baselines = self.locations[:self.num_agents].unsqueeze(1)
        self.observations = self.locations.unsqueeze(0) - agent_baselines

    """
    Updates game state given all movements and utterances and returns accrued reward
        - movements: [num_agents, config.movement_size]
        - utterances: [num_agents, config.utterance_size]
        - goal_predictions: [num_agents, num_agents, config.goal_size]
    Returns:
        - scalar: reward received in this episode of the game
    """
    def forward(self, movements, utterances, goal_predictions):
        self.locations[:self.num_agents,:] = self.locations[:self.num_agents,:] + movements
        agent_baselines = self.locations[:self.num_agents].unsqueeze(1)
        self.observations = self.locations.unsqueeze(0)- agent_baselines
        self.utterances = utterances
        return self.compute_reward(movements, goal_predictions)

    def compute_reward(self, movements, goal_predictions):
        physical_reward = self.compute_physical_reward()
        goal_pred_reward = self.compute_goal_pred_reward(goal_predictions)
        utterance_cost = self.compute_utterance_cost()
        movement_cost = self.compute_movement_cost(movements)

    """
    Computes the total reward agents get from being near their goals
    agent locations are stored as [num_agents + num_landmarks, entity_embed_size]
    """
    def compute_physical_reward(self):
        sorted_goals = goals[torch.sort(goals[:,2])[1]][:,:2]
        # [num_agents x 2] -> each agent's goal location
        return -torch.sum(
                torch.sqrt(
                    torch.sum(
                        torch.pow(self.locations[:self.num_agents,:] - sorted_goals, 2))))

    """
    Computes the total reward agents get from predicting others' goals
    """
    def compute_goal_pred_reward(self, goal_predictions):
        return 0

    """
    Computes the total cost agents get from uttering
    """
    def compute_utterance_cost(self):
        return -torch.sqrt(torch.sum(torch.pow(self.utterances,2)))

    """
    Computes the total cost agents get from moving
    """
    def compute_movement_cost(self, movements):
        return -torch.sqrt(torch.sum(torch.pow(movements,2)))


"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total reward all agents collected over the entire game
"""
class AgentModule(nn.Module):
    def __init__(self, config):
        super(AgentModule, self).__init__()
        self.game_config = config.game_config

        #self.batch_size = config.batch_size
        self.time_horizon = config.time_horizon
        self.utterance_processor = ProcessingModule(config.processing)
        self.utterance_pooling = nn.AdaptiveAvgPool1d(config.feat_vec_size)
        self.physical_processor = ProcessingModule(config.processing)
        self.physical_pooling = nn.AdaptiveAvgPool1d(config.feat_vec_size)
        self.action_processor = ActionModule(config.action)
        self.total_reward = Variable((1))

    def forward(self, game):
        for t in range(self.time_horizon):
            movements = torch.zeros((game.num_entities, config.movement_size))
            utterances = Tensor((game.num_agents, config.vocab_size))
            goal_predictions = Tensor((game.num_agents, game.num_agents, config.goal_size))
            #utterance_processes = Tensor((game.num_agents, game.num_agents, config.processing.hidden_size))
            #physical_processes = Tensor((game.num_agents, game.num_entities, config.processing.hidden_size))
            for agent in range(game.num_agents):
                utterance_processes = Tensor((game.num_agents, config.processing.hidden_size))
                physical_processes = Tensor((game.num_entities, config.processing.hidden_size))
                goal_predictions = Tensor((game.num_agents, config.goal_size))
                for other_agent in range(game.num_agents):
                    utterance_processed, game.memories.utterance[agent, other_agent], goal_predicted = self.utterance_processor(game.utterances[other_agent], game.memories.utterance[agent, other_agent])
                    #utterance_processes[agent, other_agent, :] = utterance_processed
                    utterance_processes[other_agent, :] = utterance_processed
                    goal_predictions[agent, other_agent, :] = goal_predicted

                    physical_processed, game.memories.physical[agent, other_agent] = self.physical_processor(torch.cat((game.locations[other_agent],game.physical[other_agent])), game.memories.physical[agent, other_agent])
                    #physical_processes[agent, other_agent, :] = physical_processed
                    physical_processes[other_agent, :] = physical_processed
                for landmark in range(game.num_agents, game.num_agents + game.num_landmarks):
                    physical_processed, game.memories.physical[agent, landmark] = self.physical_processor(torch.cat((game.locations[landmark],game.physical[landmark])), game.memories.physical[agent, landmark])
                    #physical_processes[agent, landmark, :] = physical_processed
                    physical_processes[landmark, :] = physical_processed

                #utterance_feat = self.utterance_pooling(utterance_processes[agent])
                #physical_feat = self.physical_pooling(physical_processes[agent])
                utterance_feat = self.utterance_pooling(utterance_processes)
                physical_feat = self.physical_pooling(physical_processes)

                movement, utterance, game.memories.action[agent] = self.action_processor(utterance_feat, physical_feat, game.goals[agent], game.memories.action[agent])
                self.movements[agent,:] = movement
                self.utterances[agent, :] = utterance

            reward = game(agent, movements, utterances, goal_predictions)
            self.total_reward += reward
        return self.total_reward

