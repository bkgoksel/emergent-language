import torch
import torch.nn as nn
import numpy as np
from typing import List, NamedTuple, Any
from torch import Tensor
from torch.autograd import Variable
import constants
from configs import ProcessingModuleConfig, GoalPredictingProcessingModuleConfig, ActionModuleConfig, AgentModuleConfig, WordCountingModuleConfig
import pdb


"""
    A Processing module takes an input from a stream and the independent memory
    of that stream and runs a single timestep of a GRU cell, followed by
    dropout and finally a linear ELU layer on top of the GRU output.
    It returns the output of the fully connected layer as well as the update to
    the independent memory.
"""
class ProcessingModule(nn.Module):
    def __init__(self, config: ProcessingModuleConfig) -> None:
        super(ProcessingModule, self).__init__()
        self.cell = nn.GRUCell(config.input_size, config.hidden_size)
        self.fully_connected = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ELU())

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
    def __init__(self, config: GoalPredictingProcessingModuleConfig) -> None:
        super(GoalPredictingProcessingModule, self).__init__()
        self.processor = ProcessingModule(config.processor)
        self.goal_predictor = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.processor.hidden_size, config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ELU(),
                nn.Linear(config.hidden_size, config.goal_size))

    def forward(self, x, mem):
        processed, mem = self.processor(x, mem)
        goal_prediction = self.goal_predictor(processed)
        return processed, mem, goal_prediction

class GumbelSoftmax(nn.Module):
    def __init__(self) -> None:
        super(GumbelSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.temp = 1

    def forward(self, x):
        U = Variable(torch.rand(x.size()))
        y = x -torch.log(-torch.log(U + 1e-20) + 1e-20)
        return self.softmax(y/self.temp)

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
    def __init__(self, config: ActionModuleConfig) -> None:
        super(ActionModule, self).__init__()
        self.using_utterances = config.use_utterances
        self.goal_processor = ProcessingModule(config.goal_processor)
        self.processor = ProcessingModule(config.action_processor)
        self.movement_step_size = config.movement_step_size
        self.movement_chooser = nn.Sequential(
                nn.Linear(config.action_processor.hidden_size, config.action_processor.hidden_size),
                nn.ELU(),
                nn.Linear(config.action_processor.hidden_size, config.movement_dim_size),
                nn.Tanh())

        if self.using_utterances:
            self.utterance_chooser = nn.Sequential(
                    nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                    nn.ELU(),
                    nn.Linear(config.hidden_size, config.vocab_size))
            self.gumbel_softmax = GumbelSoftmax()

    def forward(self, physical, goal, mem, training, utterance=None):
        goal_processed, _ = self.goal_processor(goal, mem)
        if self.using_utterances:
            x = torch.cat([physical.squeeze(1), utterance.squeeze(1), goal_processed], 1).squeeze(1)
        else:
            x = torch.cat([physical.squeeze(0), goal_processed], 1).squeeze(1)
        processed, mem = self.processor(x, mem)
        movement = self.movement_chooser(processed)
        if self.using_utterances:
            utter = self.utterance_chooser(processed)
            if training:
                utterance = self.gumbel_softmax(utter)
            else:
                utterance = torch.zeros(utter.size())
                max_utter = utter.max(1)[1]
                max_utter = max_utter.data[0]
                utterance[0, max_utter] = 1
        final_movement = (movement * 2 * self.movement_step_size) - self.movement_step_size
        if self.using_utterances:
            return final_movement, utterance, mem
        else:
            return final_movement, mem

"""
    The GameModule takes in all actions(movement, utterance, goal prediction)
    of all agents for a given timestep and returns the total cost for that
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

        config needs: -batch_size, -using_utterances, -world_dim, -vocab_size, -memory_size, -num_colors -num_shapes
"""

class GameModule(nn.Module):

    def __init__(self, config, num_agents, num_landmarks) -> None:
        super(GameModule, self).__init__()

        self.batch_size = config.batch_size # scalar: num games in this batch
        self.using_utterances = config.use_utterances # bool: whether current batch allows utterances
        self.num_agents = num_agents # scalar: number of agents in this batch
        self.num_landmarks = num_landmarks # scalar: number of landmarks in this batch
        self.num_entities = self.num_agents + self.num_landmarks # type: int

        # [batch_size, num_entities, 2]
        self.locations = Variable(torch.rand(self.batch_size, self.num_entities, 2) * config.world_dim)

        colors = (torch.rand(self.batch_size, self.num_entities, 1) * config.num_colors).floor()
        shapes = (torch.rand(self.batch_size, self.num_entities, 1) * config.num_shapes).floor()

        # [batch_size, num_entities, 2]
        self.physical = Variable(torch.cat((colors,shapes), 2).float())

        goal_agents = Tensor(self.batch_size, self.num_agents, 1)
        for b in range(self.batch_size):
            if self.num_agents == 2:
                goal_agents[b] = Tensor([1,0])
            else:
                goal_agents[b] = torch.randperm(self.num_agents)
        goal_entities = (torch.rand(self.batch_size, self.num_agents, 1) * self.num_landmarks).floor().long() + self.num_agents
        goal_locations = Tensor(self.batch_size, self.num_agents, 2)
        for b in range(self.batch_size):
            goal_locations[b] = self.locations.data[b][goal_entities[b].squeeze()]

        # [batch_size, num_agents, 3]
        self.goals = Variable(torch.cat((goal_locations, goal_agents), 2))
        goal_agents = Variable(goal_agents)


        self.memories = {
            "physical": Variable(torch.zeros(self.batch_size, self.num_agents, self.num_entities, config.memory_size)),
            "action": Variable(torch.zeros(self.batch_size, self.num_agents, config.memory_size))}

        if self.using_utterances:
            self.utterances = Variable(torch.zeros(self.batch_size, self.num_agents, config.vocab_size))
            self.memories["utterance"] = Variable(torch.zeros(self.batch_size, self.num_agents, self.num_agents, config.memory_size ))

        agent_baselines = self.locations[:, :self.num_agents, :]
        # [batch_size, num_agents, num_entities, 2]
        self.observations = self.locations.unsqueeze(1) - agent_baselines.unsqueeze(2)

        new_obs = self.goals[:,:,:2] - agent_baselines

        # [batch_size, num_agents, 2] [batch_size, num_agents, 1]
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)

    """
    Updates game state given all movements and utterances and returns accrued cost
        - movements: [batch_size, num_agents, config.movement_size]
        - utterances: [batch_size, num_agents, config.utterance_size]
        - goal_predictions: [batch_size, num_agents, num_agents, config.goal_size]
    Returns:
        - scalar: total cost of all games in the batch
    """
    def forward(self, movements, goal_predictions, utterances=None):
        self.locations = self.locations + movements
        agent_baselines = self.locations[:, :self.num_agents]
        self.observations = self.locations.unsqueeze(1)- agent_baselines.unsqueeze(2)
        new_obs = self.goals[:,:,:2] - agent_baselines
        goal_agents = self.goals[:,:,2].unsqueeze(2)
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)
        if self.using_utterances:
            self.utterances = utterances
            return self.compute_cost(movements, goal_predictions, utterances)
        else:
            return self.compute_cost(movements, goal_predictions)

    def compute_cost(self, movements, goal_predictions, utterances=None):
        physical_cost = self.compute_physical_cost()
        movement_cost = self.compute_movement_cost(movements)
        #goal_pred_cost = self.compute_goal_pred_cost(goal_predictions)
        return physical_cost + movement_cost
        """
        if self.using_utterances:
            utterance_cost = self.compute_utterance_cost()
            return physical_cost + movement_cost + goal_pred_cost + utterance_cost
        else:
            return physical_cost + goal_pred_cost + movement_cost
        """

    """
    Computes the total cost agents get from being near their goals
    agent locations are stored as [batch_size, num_agents + num_landmarks, entity_embed_size]
    """
    def compute_physical_cost(self):
        sort_idxs = torch.sort(self.goals[:,:,2])[1]
        sorted_goals = Variable(Tensor(self.goals.size()))
        for b in range(self.batch_size):
            sorted_goals[b] = self.goals[b][sort_idxs[b]]
        sorted_goals = sorted_goals[:,:,:2]
        return 2*torch.sum(
                    torch.sqrt(
                        torch.sum(
                            torch.pow(
                                self.locations[:,:self.num_agents,:] - sorted_goals, 
                                2),
                            -1)
                        )
                    )

    """
    Computes the total cost agents get from predicting others' goals
    """
    def compute_goal_pred_cost(self, goal_predictions):
        return 0

    """
    Computes the total cost agents get from uttering
    def compute_utterance_cost(self):
        return torch.sqrt(torch.sum(torch.pow(self.utterances,2)))
    """

    """
    Computes the total cost agents get from moving
    """
    def compute_movement_cost(self, movements):
        return torch.sum(torch.sqrt(torch.sum(torch.pow(movements, 2), -1)))

class WordCountingModule(nn.Module):
    def __init__(self, config: WordCountingModuleConfig) -> None:
        super(WordCountingModule, self).__init__()
        self.oov_prob = config.oov_prob
        self.word_counts = Variable(Tensor(config.vocab_size))

    def forward(self, utterances):
        cost = -utterances/(self.oov_prob + self.word_counts.sum() - 1)
        self.word_counts = self.word_counts() + utterances
        return cost


"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total cost all agents collected over the entire game
"""
class AgentModule(nn.Module):
    def __init__(self, config: AgentModuleConfig) -> None:
        super(AgentModule, self).__init__()
        self.training = True
        self.using_utterances = config.use_utterances
        self.penalizing_words = config.penalize_words
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.goal_size = config.goal_size
        self.processing_hidden_size = config.physical_processor.hidden_size
        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveAvgPool2d((1,config.feat_vec_size))
        self.action_processor = ActionModule(config.action_processor)
        self.total_cost = Variable(torch.zeros(1))

        if self.using_utterances:
            self.utterance_processor = GoalPredictingProcessingModule(config.utterance_processor)
            self.utterance_pooling = nn.AdaptiveAvgPool2d((1,config.feat_vec_size))
            if self.penalizing_words:
                self.word_counter = WordCountingModule(config.word_counter)

    def reset(self):
        self.total_cost = Variable(torch.zeros(1))

    def train(self, mode=True):
        super(AgentModule, self).train(mode)
        self.training = mode

    def update_mem(self, game, mem_str, new_mem, agent, other_agent=None):
        new_big_mem = Variable(Tensor(game.memories[mem_str].data))
        if other_agent is not None:
            new_big_mem[:, agent, other_agent] = new_mem
        else:
            new_big_mem[:, agent] = new_mem
        game.memories[mem_str] = new_big_mem

    def forward(self, game):
        if not self.training:
            timesteps = [] # type: List[Any]
        for t in range(self.time_horizon):
            movements = Variable(torch.zeros((game.batch_size, game.num_entities, self.movement_dim_size)))

            if self.using_utterances:
                utterances = Variable(Tensor(game.batch_size, game.num_agents, self.vocab_size))

            goal_predictions = Variable(Tensor(game.batch_size, game.num_agents, game.num_agents, self.goal_size))

            for agent in range(game.num_agents):
                if self.using_utterances:
                    utterance_processes = Variable(Tensor(game.batch_size, game.num_agents, self.processing_hidden_size))

                physical_processes = Variable(Tensor(game.batch_size, game.num_entities, self.processing_hidden_size))

                for other_agent in range(game.num_agents):
                    if self.using_utterances:
                        utterance_processed, new_mem, goal_predicted = self.utterance_processor(game.utterances[:,other_agent], game.memories["utterance"][:, agent, other_agent])
                        self.update_mem(game, "utterance", new_mem, agent, other_agent)
                        utterance_processes[:, other_agent, :] = utterance_processed
                        goal_predictions[:, agent, other_agent, :] = goal_predicted

                    physical_processed, new_mem = self.physical_processor(torch.cat((game.observations[:,agent,other_agent],game.physical[:,other_agent]), 1), game.memories["physical"][:,agent, other_agent])
                    self.update_mem(game, "physical", new_mem,agent, other_agent)
                    physical_processes[:,other_agent,:] = physical_processed

                for landmark in range(game.num_agents, game.num_agents + game.num_landmarks):
                    physical_processed, new_mem = self.physical_processor(torch.cat((game.observations[:,agent,landmark],game.physical[:,landmark]),1), game.memories["physical"][:,agent, landmark])
                    self.update_mem(game, "physical", new_mem, agent, landmark)
                    physical_processes[:,landmark,:] = physical_processed

                physical_feat = self.physical_pooling(physical_processes)
                if self.using_utterances:
                    utterance_feat = self.utterance_pooling(utterance_processes)
                    movement, utterance, new_mem = self.action_processor(physical_feat, game.observed_goals[:,agent], game.memories["action"][:,agent], self.training, utterance_feat)
                else:
                    movement, new_mem = self.action_processor(physical_feat, game.observed_goals[:,agent], game.memories["action"][:,agent], self.training)

                self.update_mem(game, "action", new_mem, agent)
                movements[:,agent,:] = movement

                if self.using_utterances:
                    utterances[:,agent,:] = utterance

            if self.using_utterances:
                cost = game(movements, goal_predictions, utterances)
                if self.penalizing_words:
                    cost = cost + self.word_counter(utterances)
            else:
                cost = game(movements, goal_predictions)
            self.total_cost = self.total_cost + cost
            if not self.training:
                timesteps.append({
                    'locations': game.locations,
                    'movements': movements,
                    'loss': cost})
                if self.using_utterances:
                    timesteps[-1]['utterances'] = utterances
        if self.training:
            return self.total_cost
        else:
            return self.total_cost, timesteps

