import pdb
import torch
import torch.optim as optim
from model import AgentModule
from simple_model import SimpleAgentModule
from game import Game
from configs import default_agent_module_config, default_game_config, default_training_config

def analyze_movement(moves):
    all_moves = torch.stack(moves).squeeze(1)
    print("Max move x: %f" % all_moves[:,0].max())
    print("Mean move x: %f" % all_moves[:,0].mean())
    print("Min move x: %f" % all_moves[:,0].min())
    print("Max move y: %f" % all_moves[:,1].max())
    print("Mean move y: %f" % all_moves[:,1].mean())
    print("Min move y: %f" % all_moves[:,1].min())

def main():
    agent = AgentModule(default_agent_module_config)
    #agent = SimpleAgentModule()
    optimizer = optim.RMSprop(agent.parameters(), lr=default_training_config.learning_rate)
    #running_costs = []
    for epoch in range(default_training_config.num_epochs):
        agent.reset()
        game = Game(default_game_config).get_vectorized_state()
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

