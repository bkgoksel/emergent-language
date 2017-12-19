import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.processing import ProcessingModule
from modules.goal_predicting import GoalPredictingProcessingModule
from modules.action import ActionModule
from modules.word_counting import WordCountingModule


"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total cost all agents collected over the entire game
"""
class AgentModule(nn.Module):
    def __init__(self, config):
        super(AgentModule, self).__init__()
        self.training = True
        self.using_utterances = config.use_utterances
        self.penalizing_words = config.penalize_words
        self.using_cuda = config.use_cuda
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.goal_size = config.goal_size
        self.processing_hidden_size = config.physical_processor.hidden_size
        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveMaxPool2d((1,config.feat_vec_size))
        self.action_processor = ActionModule(config.action_processor)

        if self.using_cuda:
            self.total_cost = Variable(torch.zeros(1).cuda())
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.total_cost = Variable(torch.zeros(1))
            self.Tensor = torch.FloatTensor

        if self.using_utterances:
            self.utterance_processor = GoalPredictingProcessingModule(config.utterance_processor)
            self.utterance_pooling = nn.AdaptiveMaxPool2d((1,config.feat_vec_size))
            if self.penalizing_words:
                self.word_counter = WordCountingModule(config.word_counter)

    def reset(self):
        total_cost = torch.zeros(1).cuda() if self.using_cuda else torch.zeros(1)
        self.total_cost = Variable(total_cost)
        if self.using_utterances and self.penalizing_words:
            self.word_counter.word_counts = Variable(self.Tensor(self.vocab_size))

    def train(self, mode=True):
        super(AgentModule, self).train(mode)
        self.training = mode

    def update_mem(self, game, mem_str, new_mem, agent, other_agent=None):
        new_big_mem = Variable(self.Tensor(game.memories[mem_str].data))
        if other_agent is not None:
            new_big_mem[:, agent, other_agent] = new_mem
        else:
            new_big_mem[:, agent] = new_mem
        game.memories[mem_str] = new_big_mem

    def forward(self, game):
        if not self.training:
            timesteps = [] # type: List[Any]
        for t in range(self.time_horizon):
            movements = torch.zeros((game.batch_size, game.num_entities, self.movement_dim_size))
            movements = movements.cuda() if self.using_cuda else movements
            movements = Variable(movements)
            if self.using_utterances:
                utterances = Variable(self.Tensor(game.batch_size, game.num_agents, self.vocab_size))
            goal_predictions = Variable(self.Tensor(game.batch_size, game.num_agents, game.num_agents, self.goal_size))

            for agent in range(game.num_agents):
                if self.using_utterances:
                    utterance_processes = Variable(self.Tensor(game.batch_size, game.num_agents, self.processing_hidden_size))
                physical_processes = Variable(self.Tensor(game.batch_size, game.num_entities, self.processing_hidden_size))

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

