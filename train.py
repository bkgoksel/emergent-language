import argparse
import pdb
import torch
import torch.optim as optim
import configs
from model import AgentModule
from simple_model import SimpleAgentModule
from game import Game

parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
parser.add_argument('--no-utterances', '--nu', action='store_true', help='if specified disables the communications channel (default enabled)')
parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
parser.add_argument('--learning-rate', type=int, help='if specified sets learning rate (default 1e-3)')
parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default 32)')
parser.add_argument('--n-agents', '-a', type=int, help='if specified sets maximum number of agents in each episode (default 3)')
parser.add_argument('--n-landmarks', '-l', type=int, help='if specified sets maximum number of landmarks in each episode (default 3)')
parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode (default 6)')

def analyze_movement(moves):
    all_moves = torch.stack(moves).squeeze(1)
    print("Max move x: %f" % all_moves[:,0].max())
    print("Mean move x: %f" % all_moves[:,0].mean())
    print("Min move x: %f" % all_moves[:,0].min())
    print("Max move y: %f" % all_moves[:,1].max())
    print("Mean move y: %f" % all_moves[:,1].mean())
    print("Min move y: %f" % all_moves[:,1].min())

def main():
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)
    agent = AgentModule(agent_config)
    #agent = SimpleAgentModule()
    optimizer = optim.RMSprop(agent.parameters(), lr=training_config.learning_rate)
    #running_costs = []
    for epoch in range(training_config.num_epochs):
        agent.reset()
        game = Game(game_config).get_vectorized_state()
        optimizer.zero_grad()
        total_loss = agent(game)
        #analyze_movement(agent.all_movements)
        total_loss.backward(retain_graph=True)
        #pdb.set_trace()
        optimizer.step()
        #running_costs.append(total_loss)
        print("[Ep: %d][Loss: %f]" % (epoch, total_loss.data[0]))

if __name__ == "__main__":
    main()

