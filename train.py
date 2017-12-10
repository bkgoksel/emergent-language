import argparse
import numpy as np
import torch
import torch.optim as optim
import configs
from model import AgentModule, GameModule
from game import Game
from collections import defaultdict

parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
parser.add_argument('--no-utterances', action='store_true', help='if specified disables the communications channel (default enabled)')
parser.add_argument('--penalize-words', action='store_true', help='if specified penalizes uncommon word usage (default disabled)')
parser.add_argument('--self-goals', action='store_true', help='if specified each agent always gets its own goal (default disabled)')
parser.add_argument('--use-random-colors', action='store_true', help='if specified gives random colors(RGB) to object rather than 3 predetermined colors (default disabled, each object is R,G or B)')
parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
parser.add_argument('--learning-rate', type=int, help='if specified sets learning rate (default 1e-3)')
parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default 32)')
parser.add_argument('--max-agents', type=int, help='if specified sets maximum number of agents in each episode (default 3)')
parser.add_argument('--min-agents', type=int, help='if specified sets minimum number of agents in each episode (default 1)')
parser.add_argument('--max-landmarks', type=int, help='if specified sets maximum number of landmarks in each episode (default 3)')
parser.add_argument('--min-landmarks', type=int, help='if specified sets maximum number of landmarks in each episode (default 3)')
parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode (default 6)')
parser.add_argument('--world-dim', '-w', type=int, help='if specified sets the side length of the square grid where all agents and landmakrs spawn(default 16)')
parser.add_argument('--oov-prob', '-o', type=int, help='higher value penalize uncommon words less when penalizing words (default 6)')
parser.add_argument('--load-model-weights', type=str, help='if specified start with saved model weights saved at file given by this argument')
parser.add_argument('--save-model-weights', type=str, help='if specified save the model weights at file given by this argument')

def print_losses_for(past, epoch, running_costs, losses, game_config):
    print("[epoch: %d][Last %d eps avg cost: %f]" % (epoch, past+1, np.mean(running_costs[-past:])))
    for a in range(game_config.min_agents, game_config.max_agents + 1):
        for l in range(game_config.min_landmarks, game_config.max_landmarks + 1):
            print("[Last %d eps avg cost for [%d agents, %d landmarks]: %f]" % (past+1, a, l, np.mean(losses[a][l][-past:])))


def print_losses(epoch, running_costs, losses, game_config):
    if epoch % 10 == 9:
        print_losses_for(9, epoch, running_costs, losses, game_config)
    if epoch % 100 == 99:
        print_losses_for(99, epoch, running_costs, losses, game_config)

def main():
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)
    print("Training with config:")
    print(training_config)
    print(game_config)
    print(agent_config)
    agent = AgentModule(agent_config)
    optimizer = optim.RMSprop(agent.parameters(), lr=training_config.learning_rate)
    running_costs = []
    losses = defaultdict(lambda:defaultdict(list))
    for epoch in range(training_config.num_epochs):
        agent.reset()
        game = GameModule.from_structured_game(Game(game_config))
        optimizer.zero_grad()
        total_loss = agent(game)
        total_loss.backward()
        optimizer.step()
        running_costs.append(total_loss)
        losses[game.num_agents][game.num_landmarks].append(total_loss.data[0])
        print_losses(epoch, running_costs, losses, game_config)
    if training_config.save_model:
        torch.save(agent, training_config.save_model_file)
        print("Saved agent model weights at %s" % training_config.save_model_file)
    import code
    code.interact(local=locals())


if __name__ == "__main__":
    main()

