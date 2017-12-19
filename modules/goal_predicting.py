import torch.nn as nn
from modules.processing import ProcessingModule

"""
    A GoalPredictingProcessingModule acts like a regular processing module but
    also runs a goal predictor layer that is a two layer fully-connected
    network. It returns the regular processing module's output, its memory
    update and finally a goal vector sized goal prediction
"""
class GoalPredictingProcessingModule(nn.Module):
    def __init__(self, config):
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
