from typing import Tuple, NamedTuple, List, Optional, Set, Any
import pdb
import numpy as np # type: ignore
import constants
from configs import GameConfig

Goal = NamedTuple('Goal', [('location', Any), ('agent_id', int)])

Landmark = NamedTuple('Landmark', [
    ('color', Any),
    ('location', Any),
    ('shape', Optional[int])
])

Agent = NamedTuple('Agent', [
    ('color', Any),
    ('location', Any),
    ('shape', Optional[int]),
    ('goal', Goal)
])

GameState = NamedTuple('GameState', [
    ('agents', List[Agent]),
    ('landmarks', List[Landmark]),
    ('num_agents', int),
    ('num_landmarks', int)
])

class Game():
    @classmethod
    def from_vectorized_game(cls, vec_game):
        config = GameConfig(
                [0,0],
                vec_game.num_agents,
                vec_game.locations.shape[0] - vec_game.num_agents,
                0,
                False,
                [],
                True,
                [],
                True,
                vec_game.utterances.shape[1],
                vec_game.memories.action.shape[1],
                False)
        agents = [] # type: List[Agent]
        for i in range(vec_game.num_agents):
            agents.append(
                    Agent(
                        vec_game.locations[i,:],
                        vec_game.physical[i,:3],
                        vec_game.physical[i,3],
                        Goal(
                            vec_game.goals[i,:2],
                            vec_game.goals[i,2]
                            )
                        )
                    )
        landmarks = [] # type: List[Landmark]
        for i in range(vec_game.num_agents, vec_game.locations.shape[0]):
            landmarks.append(
                    Landmark(
                        vec_game.locations[i,:],
                        vec_game.physical[i,:3],
                        vec_game.physical[i,3]
                        )
                    )
        state = GameState(agents, landmarks)
        return cls(config, state)

    def __init__(self, game_config: GameConfig, init_state: GameState=None) -> None:
        self.config = game_config # type: GameConfig
        self.occupied_locations = set() # type: Set[Any]
        self.state = init_state # type: GameState
        if init_state is None:
            self.initialize_random_state()

    def initialize_random_state(self) -> None:
        num_landmarks = np.random.randint(self.config.min_landmarks, self.config.max_landmarks + 1)
        num_agents = np.random.randint(self.config.min_agents, self.config.max_agents + 1)
        self.state = GameState([],[], num_landmarks=num_landmarks, num_agents=num_agents)
        self.init_random_landmarks()
        self.init_random_agents()

    def get_random_color(self):
        if self.config.use_strict_colors:
            return self.config.strict_colors[np.random.randint(self.config.strict_colors.shape[0])]
        else:
            return np.random.randint(constants.COLOR_SCALE, size=3)

    def get_random_location(self) -> Any:
        return np.multiply(np.random.rand(2), self.config.world_dim)

    def init_random_landmarks(self) -> None:
        for i in range(self.state.num_landmarks):
            c = self.get_random_color()
            l = self.get_random_location()
            s = np.random.randint(self.config.num_shapes) if self.config.use_shapes else None
            self.state.landmarks.append(Landmark(c,l,s))

    def init_random_agents(self) -> None:
        goal_order = list(range(self.state.num_agents)) # type: List[int]
        if not self.config.self_goals:
            np.random.shuffle(goal_order)
        for i in range(self.state.num_agents):
            c = self.get_random_color()
            l = self.get_random_location()
            s = np.random.randint(self.config.num_shapes) if self.config.use_shapes else None
            g = Goal(self.state.landmarks[np.random.randint(self.state.num_landmarks)].location, goal_order[i])
            self.state.agents.append(Agent(c, l, s, g))

