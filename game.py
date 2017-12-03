from typing import Tuple, NamedTuple, List, Optional, Set, Any
import numpy as np # type: ignore
import constants
from model import VectorizedGame

Location = NamedTuple('Location', [('x', float), ('y', float)])
Color = NamedTuple('Color', [('r', int), ('g', int), ('b', int)])
Goal = NamedTuple('Goal', [('agent_id', int), ('location', Location)])

GameConfig = NamedTuple('GameConfig', [
    ('world_dim', 'Location'),
    ('num_agents', int),
    ('num_landmarks', int),
    ('num_timesteps', int),
    ('use_strict_colors', bool),
    ('strict_colors', List[Color]),
    ('use_shapes', bool),
    ('shapes', List[int]),
    ('vocab_size', int),
    ('memory_size', int)
])

Landmark = NamedTuple('Landmark', [
    ('color', Color),
    ('location', Location),
    ('shape', Optional[int])
])

Agent = NamedTuple('Agent', [
    ('color', Color),
    ('location', Location),
    ('shape', Optional[int]),
    ('goal', Goal)
])

GameState = NamedTuple('GameState', [
    ('agents', List[Agent]),
    ('landmarks', List[Landmark])
])



class Game():

    @classmethod
    def from_vectorized_game(cls, vec_game):
        config = GameConfig(
                Location(0,0),
                vec_game.num_agents,
                vec_game.physical.shape[0] - vec_game.num_agents,
                0,
                False,
                [],
                True,
                [],
                vec_game.utterances.shape[1],
                vec_game.memories.action.shape[1])
        agents = [] # type: List[Agent]
        for i in range(vec_game.num_agents):
            agents.append(
                    Agent(
                        Location(vec_game.physical[i,0], vec_game.physical[i,1]),
                        Color(vec_game.physical[i,2], vec_game.physical[i,3], vec_game.physical[i,4]),
                        vec_game.physical[i,5],
                        Goal(
                            Location(vec_game.goals[i,0], vec_game.goals[i,1]),
                            vec_game.goals[i,2]
                            )
                        )
                    )
        landmarks = [] # type: List[Landmark]
        for i in range(vec_game.num_agents, vec_game.physical.shape[0]):
            landmarks.append(
                    Landmark(
                        Location(vec_game.physical[i,0], vec_game.physical[i,1]),
                        Color(vec_game.physical[i,2], vec_game.physical[i,3], vec_game.physical[i,4]),
                        vec_game.physical[i,5]
                        )
                    )
        state = GameState(agents, landmarks)
        return cls(config, state)

    def __init__(self, game_config: GameConfig, init_state: GameState=None) -> None:
        self.config = game_config # type: GameConfig
        self.occupied_locations = set() # type: Set[Location]
        self.state = init_state # type: GameState
        if init_state is None:
            self.initialize_random_state()

    def initialize_random_state(self) -> None:
        self.state = GameState([],[])
        self.init_random_landmarks()
        self.init_random_agents()

    def get_random_color(self) -> Color:
        if self.config.use_strict_colors:
            return np.random.choice(self.config.strict_colors)
        else:
            return Color(np.random.rand() * constants.COLOR_SCALE,
                    np.random.rand() * constants.COLOR_SCALE,
                    np.random.rand() * constants.COLOR_SCALE)

    def get_random_location(self) -> Location:
        l = Location(np.random.rand() * self.config.world_dim.x, np.random.rand() * self.config.world_dim.y)
        while l in self.occupied_locations:
            l = Location(np.random.rand() * self.config.world_dim.x, np.random.rand() * self.config.world_dim.y)
        self.occupied_locations.add(l)
        return l

    def init_random_landmarks(self) -> None:
        for i in range(self.config.num_landmarks):
            c = self.get_random_color()
            l = self.get_random_location()
            s = np.random.choice(self.config.shapes) if self.config.use_shapes else None
            self.state.landmarks.append(Landmark(c,l,s))

    def init_random_agents(self) -> None:
        goal_order = list(range(self.config.num_agents)) # type: List[int]
        np.random.shuffle(goal_order)
        for i in range(self.config.num_agents):
            c = self.get_random_color()
            l = self.get_random_location()
            s = np.random.choice(self.config.shapes) if self.config.use_shapes else None
            g = Goal(goal_order[i], np.random.choice(self.state.landmarks).location)
            self.state.agents.append(Agent(c, l, s, g))

    def get_vectorized_state(self) -> VectorizedGame:
        agents = np.empty([self.config.num_agents, constants.AGENT_EMBED_DIM])
        landmarks = np.empty([self.config.num_landmarks, constants.LANDMARK_EMBED_DIM])
        goals = np.empty([self.config.num_agents, constants.GOAL_EMBED_DIM])

        for i, agent in enumerate(self.state.agents):
            agents[i][0] = agent.location.x
            agents[i][1] = agent.location.y
            agents[i][2] = agent.color.r
            agents[i][3] = agent.color.g
            agents[i][4] = agent.color.b
            agents[i][5] = agent.shape
            goals[i][0] = agent.goal.location.x
            goals[i][1] = agent.goal.location.y
            goals[i][2] = agent.goal.agent_id

        for i, landmark in enumerate(self.state.landmarks):
            landmarks[i][0] = landmark.location.x
            landmarks[i][1] = landmark.location.y
            landmarks[i][2] = landmark.color.r
            landmarks[i][3] = landmark.color.g
            landmarks[i][4] = landmark.color.b
            landmarks[i][5] = landmark.shape

        return VectorizedGame(agents, landmarks, goals, self.config.vocab_size, self.config.memory_size)
