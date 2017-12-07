import torch
import torch.nn as nn
import numpy as np
from typing import NamedTuple, Any
from torch import Tensor
from torch.autograd import Variable
import constants
from configs import ProcessingModuleConfig, GoalPredictingProcessingModuleConfig, ActionModuleConfig, AgentModuleConfig
import pdb


"""
    A Processing module takes an input from a stream and the independent memory
    of that stream and runs a single timestep of a GRU cell, followed by
    dropout and finally a linear ReLU layer on top of the GRU output.
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
                nn.ReLU())

    def forward(self, x, m):
        m = self.cell(x.unsqueeze(0), m.unsqueeze(0))
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
                nn.Linear(config.processor.hidden_size, config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.goal_size))

    def forward(self, x, mem):
        processed, mem = self.processor(x, mem)
        goal_prediction = self.goal_predictor(processed)
        return processed, mem, goal_prediction

class GumbelSoftmax(nn.Module):
    def __init__(self) -> None:
        super(GumbelSoftmax, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        U = Variable(torch.rand(x.size()))
        y = x -torch.log(-torch.log(U + 1e-20) + 1e-20)
        return self.softmax(y)

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
        self.goal_processor = ProcessingModule(config.goal_processor)
        self.processor = ProcessingModule(config.action_processor)
        self.movement_step_size = config.movement_step_size
        self.movement_chooser = nn.Sequential(
                nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.action_processor.hidden_size, config.movement_dim_size),
                nn.Softmax()
                )
        self.utterance_chooser = nn.Sequential(
                nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.vocab_size),
                GumbelSoftmax()
                )

    def forward(self, physical, utterance, goal, mem):
        goal_processed, _ = self.goal_processor(goal, mem)
        x = torch.cat([physical.squeeze(0), utterance.squeeze(0), goal_processed], 1).squeeze(0)
        processed, mem = self.processor(x, mem)
        movement = self.movement_chooser(processed)
        utterance = self.utterance_chooser(processed)
        final_movement = torch.add(torch.mul(movement, 2*self.movement_step_size), -self.movement_step_size)
        return final_movement, utterance, mem

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
"""

class GameModule(nn.Module):
    def __init__(self, agent_locations, agent_physical, landmark_locations, landmark_physical, goals, vocab_size, memory_size) -> None:
        super(GameModule, self).__init__()
        self.num_agents = agent_locations.shape[0]
        self.num_landmarks = landmark_locations.shape[0]
        self.num_entities = self.num_agents + self.num_landmarks # type: int

        self.locations = Variable(torch.from_numpy(np.concatenate((agent_locations, landmark_locations))).float())
        self.physical = Variable(torch.from_numpy(np.concatenate((agent_physical, landmark_physical))).float())

        self.goals = Variable(torch.from_numpy(goals).float())
        self.utterances = Variable(torch.zeros(self.num_agents, vocab_size))
        self.memories = {
                "utterance": Variable(torch.zeros(self.num_agents, self.num_agents, memory_size)),
                "physical": Variable(torch.zeros(self.num_agents, self.locations.shape[0], memory_size)),
                "action": Variable(torch.zeros(self.num_agents, memory_size))}

        agent_baselines = self.locations[:self.num_agents].unsqueeze(1)
        self.observations = self.locations.unsqueeze(0) - agent_baselines

    """
    Updates game state given all movements and utterances and returns accrued cost
        - movements: [num_agents, config.movement_size]
        - utterances: [num_agents, config.utterance_size]
        - goal_predictions: [num_agents, num_agents, config.goal_size]
    Returns:
        - scalar: cost received in this episode of the game
    """
    def forward(self, movements, utterances, goal_predictions):
        new_locations = self.locations + movements
        self.locations = new_locations
        agent_baselines = self.locations[:self.num_agents].unsqueeze(1)
        self.observations = self.locations.unsqueeze(0)- agent_baselines
        self.utterances = utterances
        return self.compute_cost(movements, goal_predictions)

    def compute_cost(self, movements, goal_predictions):
        physical_cost = self.compute_physical_cost()
        goal_pred_cost = self.compute_goal_pred_cost(goal_predictions)
        utterance_cost = self.compute_utterance_cost()
        movement_cost = self.compute_movement_cost(movements)
        return physical_cost + goal_pred_cost + utterance_cost + movement_cost

    """
    Computes the total cost agents get from being near their goals
    agent locations are stored as [num_agents + num_landmarks, entity_embed_size]
    """
    def compute_physical_cost(self):
        sorted_goals = self.goals[torch.sort(self.goals[:,2])[1]][:,:2]
        # [num_agents x 2] -> each agent's goal location
        return torch.sum(
                torch.sqrt(
                    torch.sum(
                        torch.pow(self.locations[:self.num_agents,:] - sorted_goals, 2))))

    """
    Computes the total cost agents get from predicting others' goals
    """
    def compute_goal_pred_cost(self, goal_predictions):
        return 0

    """
    Computes the total cost agents get from uttering
    """
    def compute_utterance_cost(self):
        return torch.sqrt(torch.sum(torch.pow(self.utterances,2)))

    """
    Computes the total cost agents get from moving
    """
    def compute_movement_cost(self, movements):
        return torch.sqrt(torch.sum(torch.pow(movements,2)))


"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total cost all agents collected over the entire game
"""
class AgentModule(nn.Module):
    def __init__(self, config: AgentModuleConfig) -> None:
        super(AgentModule, self).__init__()
        # Save config vals that will be needed
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.goal_size = config.goal_size
        self.processing_hidden_size = config.physical_processor.hidden_size
        # Set-up the processing modules
        self.utterance_processor = GoalPredictingProcessingModule(config.utterance_processor)
        self.utterance_pooling = nn.AdaptiveAvgPool2d((1,config.feat_vec_size))
        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveAvgPool2d((1,config.feat_vec_size))
        self.action_processor = ActionModule(config.action_processor)
        # Store the total cost
        self.total_cost = Variable(torch.zeros(1))

    def reset(self):
        self.total_cost = Variable(torch.zeros(1))

    def update_mem(self, game, mem_str, new_mem, agent, other_agent=None):
        new_big_mem = Variable(Tensor(game.memories[mem_str].data))
        if other_agent is not None:
            new_big_mem[agent, other_agent] = new_mem
        else:
            new_big_mem[agent] = new_mem
        game.memories[mem_str] = new_big_mem

    def forward(self, game):
        for t in range(self.time_horizon):
            movements = Variable(torch.zeros((game.num_entities, self.movement_dim_size)))
            utterances = Variable(Tensor(game.num_agents, self.vocab_size))
            goal_predictions = Variable(Tensor(game.num_agents, game.num_agents, self.goal_size))
            for agent in range(game.num_agents):

                # Prepare the tensors to gather all processing module outputs for this agent
                utterance_processes = Variable(Tensor(game.num_agents, self.processing_hidden_size))
                physical_processes = Variable(Tensor(game.num_entities, self.processing_hidden_size))

                for other_agent in range(game.num_agents):
                    # process the utterance from this other agent
                    utterance_processed, new_mem, goal_predicted = self.utterance_processor(game.utterances[other_agent], game.memories["utterance"][agent, other_agent])
                    self.update_mem(game, "utterance", new_mem, agent, other_agent)
                    #new_big_mem = Variable(Tensor(game.memories["utterance"].data))
                    #new_big_mem[agent, other_agent] = new_mem
                    #game.memories["utterance"] = new_big_mem
                    utterance_processes[other_agent, :] = utterance_processed
                    goal_predictions[agent, other_agent, :] = goal_predicted

                    # process the physical input from this other agent
                    physical_processed, new_mem = self.physical_processor(torch.cat((game.locations[other_agent],game.physical[other_agent])), game.memories["physical"][agent, other_agent])
                    self.update_mem(game, "physical", new_mem,agent, other_agent)
                    #new_big_mem = Variable(Tensor(game.memories["physical"].data))
                    #new_big_mem[agent, other_agent] = new_mem
                    #game.memories["physical"] = new_big_mem
                    physical_processes[other_agent, :] = physical_processed

                for landmark in range(game.num_agents, game.num_agents + game.num_landmarks):
                    # process the physical input from this landmark
                    physical_processed, new_mem = self.physical_processor(torch.cat((game.locations[landmark],game.physical[landmark])), game.memories["physical"][agent, landmark])
                    self.update_mem(game, "physical", new_mem, agent, landmark)
                    physical_processes[landmark, :] = physical_processed

                # Pool the processing module outputs for an overall physical and utterance input
                utterance_feat = self.utterance_pooling(utterance_processes.unsqueeze(0))
                physical_feat = self.physical_pooling(physical_processes.unsqueeze(0))

                # Choose a move and utterance based on pooled inputs of this timestep
                movement, utterance, new_mem = self.action_processor(utterance_feat, physical_feat, game.goals[agent], game.memories["action"][agent])
                self.update_mem(game, "action", new_mem, agent)
                #new_big_mem = Variable(Tensor(game.memories["action"].data))
                #new_big_mem[agent] = new_mem
                #game.memories["action"] = new_big_mem
                # save the actions
                movements[agent,:] = movement
                #pdb.set_trace()
                utterances[agent, :] = utterance

            # Compute the cost for this timestep given all agents' chosen actions
            cost = game(movements, utterances, goal_predictions)
            self.total_cost += cost
        return self.total_cost

