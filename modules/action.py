import torch
import torch.nn as nn

from modules.processing import ProcessingModule
from modules.gumbel_softmax import GumbelSoftmax

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
        self.using_utterances = config.use_utterances
        self.using_cuda = config.use_cuda
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
            self.gumbel_softmax = GumbelSoftmax(config.use_cuda)

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
                if self.using_cuda:
                    utterance = utterance.cuda()
                max_utter = utter.max(1)[1]
                max_utter = max_utter.data[0]
                utterance[0, max_utter] = 1
        final_movement = (movement * 2 * self.movement_step_size) - self.movement_step_size
        if self.using_utterances:
            return final_movement, utterance, mem
        else:
            return final_movement, mem
