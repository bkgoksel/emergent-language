import argparse
import torch.optim as optim
import configs
from model import AgentModule
from game import Game

parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
parser.add_argument('--no-utterances', action='store_true', help='if specified disables the communications channel (default enabled)')
parser.add_argument('--penalize-words', action='store_true', help='if specified penalizes uncommon word usage (default disabled)')
parser.add_argument('--use-random-colors', action='store_true', help='if specified gives random colors(RGB) to object rather than 3 predetermined colors (default disabled, each object is R,G or B)')
parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
parser.add_argument('--learning-rate', type=int, help='if specified sets learning rate (default 1e-3)')
parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default 32)')
parser.add_argument('--n-agents', '-a', type=int, help='if specified sets maximum number of agents in each episode (default 3)')
parser.add_argument('--n-landmarks', '-l', type=int, help='if specified sets maximum number of landmarks in each episode (default 3)')
parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode (default 6)')
parser.add_argument('--world-dim', '-w', type=int, help='if specified sets the side length of the square grid where all agents and landmakrs spawn(default 16)')
parser.add_argument('--oov-prob', '-o', type=int, help='higher value penalize uncommon words less when penalizing words (default 6)')

def main():
    args = vars(parser.parse_args())
    print("Training with arguments:")
    for k,v in args.items():
        print("  - %s: %s" % (k, v))
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)
    agent = AgentModule(agent_config)
    optimizer = optim.RMSprop(agent.parameters(), lr=training_config.learning_rate)
    running_costs = []
    for epoch in range(training_config.num_epochs):
        agent.reset()
        game = Game(game_config).get_vectorized_state()
        optimizer.zero_grad()
        total_loss = agent(game)
        total_loss.backward(retain_graph=True)
        optimizer.step()
        running_costs.append(total_loss)
        print("[Ep: %d][Loss: %f]" % (epoch, total_loss.data[0]))

if __name__ == "__main__":
    main()

