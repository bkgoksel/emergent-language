import torch.optim as optim
from model import AgentModule
from game import Game
from configs import default_agent_module_config, default_game_config, default_training_config

def main():
    agent = AgentModule(default_agent_module_config)
    optimizer = optim.RMSprop(agent.parameters())
    #running_costs = []
    for epoch in range(default_training_config.num_epochs):
        agent.reset()
        game = Game(default_game_config).get_vectorized_state()
        optimizer.zero_grad()
        total_loss = agent(game)
        total_loss.backward(retain_graph=True)
        optimizer.step()
        #running_costs.append(total_loss)
        print("[Ep: %d][Loss: %f]" % (epoch, total_loss))

if __name__ == "__main__":
    main()

