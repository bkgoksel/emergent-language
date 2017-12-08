import torch
import torch.nn as nn
import numpy as np
from typing import NamedTuple, Any
from torch import Tensor
from torch.autograd import Variable
import constants
from configs import ProcessingModuleConfig, GoalPredictingProcessingModuleConfig, ActionModuleConfig, AgentModuleConfig
import pdb

class SimpleProcessingModule(nn.Module):
    def __init__(self):
        super(SimpleProcessingModule, self).__init__()
        self.layer = nn.Sequential(
                nn.Linear(6, 16),
                nn.ELU(),
                nn.Linear(16, 16))

    def forward(self, loc, physical_info):
        x = torch.cat((loc, physical_info))
        return self.layer(x)

class GoalProcessingModule(nn.Module):
    def __init__(self):
        super(GoalProcessingModule, self).__init__()
        self.layer = nn.Sequential(
                nn.Linear(3, 16),
                nn.ELU(),
                nn.Linear(16, 16))

    def forward(self, goal):
        return self.layer(goal)

class SimpleMoveModule(nn.Module):
    def __init__(self):
        super(SimpleMoveModule, self).__init__()
        self.layer = nn.Sequential(
                nn.Linear(32, 64),
                nn.ELU(),
                nn.Linear(64, 2),
                nn.Tanh())

    def forward(self, goal, physical):
        goal_and_loc = torch.cat((goal, physical))
        move = self.layer(goal_and_loc)
        move = (move * 2) - 1
        return move

class SimpleAgentModule(nn.Module):
    def __init__(self):
        super(SimpleAgentModule, self).__init__()
        self.physical_module = SimpleProcessingModule()
        self.goal_module = GoalProcessingModule()
        self.move_module = SimpleMoveModule()

    def forward(self, game):
        movements = Variable(torch.zeros((game.num_entities, 2)))

        physical_processed = self.physical_module(game.observations[0,0,:], game.physical[0])
        goal_processed = self.goal_module(game.observed_goals[0].squeeze())

        move = self.move_module(goal_processed, physical_processed)

        movements[0,:] = move
        utterances = Variable(torch.zeros(1, 32))
        goal_predictions = Variable(torch.zeros(1,1))
        loss = game(movements, utterances, goal_predictions)
        return loss

