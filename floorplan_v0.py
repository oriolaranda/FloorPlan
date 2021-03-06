# -*- coding: utf-8 -*-
"""
FloorPlan-v0
Gym Environment for Floor Planning, architecture design

"""

import gym
from gym import spaces
import glob
from PIL import Image
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause

# #### CONSTANTS ##### #

cmap = {1: {0: [[255, 255, 255], '#FFFFFF'], 127: [[0, 0, 0], '#000000'], 255: [[255, 0, 0], '#FF0000'],
            -1: [[192, 192, 192], '#C0C0C0']},
        2: {0: [[210, 105, 30], '#D2691E'], 1: [[220, 20, 60], '#DC143C'], 2: [[255, 215, 0], '#FFD700'],
            3: [[30, 144, 255], '#1E90FF'], 4: [[244, 164, 96], '#F4A460'], 5: [[255, 99, 71], '#FF6347'],
            6: [[240, 128, 128], '#F08080'], 7: [[255, 160, 122], '#FFA07A'], 8: [[255, 165, 0], '#FFA500'],
            9: [[0, 255, 127], '#00FF7F'], 10: [[255, 228, 181], '#FFE4B5'], 11: [[188, 143, 143], '#BC8F8F'],
            12: [[139, 69, 19], '#8B4513'], 13: [[255, 255, 255], '#FFFFFF'], 14: [[0, 0, 0], '#000000'],
            15: [[255, 0, 0], '#FF0000'], 16: [[128, 128, 128], '#808080'], 17: [[128, 0, 0], '#800000'],
            -1: [[192, 192, 192], '#C0C0C0']},
        4: {0: [[255, 255, 255], '#FFFFFF'], 255: [[119, 136, 153], '#778899'], -1: [[192, 192, 192], '#C0C0C0']}}


######################


class FloorPlanEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(self, config):
        super(FloorPlanEnv, self).__init__()

        # RENDER SETTINGS
        ##################
        self.window_size = config.get('window_size', 256)
        self.scale = self.window_size // 256

        # SETUP PARAMETERS
        ###################
        self.steps_per_episode = config.get('steps_per_episode', 1000)
        self.num_rooms, self.rooms_type = self._decide_rooms(config)

        # OBSERVATION SPACE
        ####################
        # Observation space for each room (Dict)
        obs_space = dict()
        for room in range(self.num_rooms):
            space = spaces.Dict({
                "pos": spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8),
                "size": spaces.Box(low=5, high=2000, shape=(1,), dtype=np.uint8),
                "proportion": spaces.Box(low=0.1, high=10, shape=(1,), dtype=np.float16)
            })
            obs_space.update({room: space})
        # Defining observation space
        self.observation_space = spaces.Dict(obs_space)

        # ACTION SPACE
        ###############
        # Mapping from discrete actions to state variable and values
        self.action_ids = {
            0: ("pos", [5, 0]),
            1: ("pos", [-5, 0]),
            2: ("pos", [0, 5]),
            3: ("pos", [0, -5]),
            4: ("size", 400),
            5: ("size", -400),
            6: ("proportion", 0.25),
            7: ("proportion", -0.25),
        }

        # Defining the action space
        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([self.num_rooms - 1, len(self.action_ids) - 1]),
                                       dtype=np.uint8)
        # Clipping action space
        self.bounds = {
            "pos": (0, 255),
            "size": (400, 2000),
            "proportion": (0.1, 10)
        }

        # INFO FROM FLOOR IMAGE
        ########################
        self.floors = self._load_floors(config['data_dir'])
        self.floor = next(self.floors)
        self.canvas_floor = self._make_canvas_floor()
        self.rows, self.cols = self._compute_floor_limits()
        self.floor_area = self._compute_floor_area()
        self.door_pos = self._compute_door_pos()

        # Initialization
        self.current_step = 0
        self.window = None
        self.clock = None
        self.current_state = None

    def _decide_rooms(self, config):
        num_rooms = config.get('num_rooms', 8)
        fixed_rooms = config.get('fixed_rooms', [0, 1, 2, 3])
        assert len(fixed_rooms) <= num_rooms, "Number of rooms is smaller than the given fixed rooms"

        rooms_left = num_rooms - len(fixed_rooms)
        optional_rooms = []
        if rooms_left > 0:
            optional_rooms = list(np.random.randint(3, 12, size=rooms_left))

        all_rooms = fixed_rooms + optional_rooms
        rooms_type = dict(zip(range(num_rooms), all_rooms))
        return num_rooms, rooms_type

    def _load_floors(self, data_dir):
        # Load the entire dataset as a generator
        files = glob.glob(data_dir + '/*.png')

        for file in files:
            im = Image.open(file)  # rgba image
            im_array = np.asarray(im)
            yield im_array

    def _compute_floor_limits(self):
        raw = self.floor[:, :, 0]
        rows, cols = [], []

        for i in range(256):
            row_points = list(np.where(raw[i, :] > 0)[0])
            col_points = list(np.where(raw[:, i] > 0)[0])
            row_interval = row_points
            col_interval = col_points

            if row_points:
                row_interval = [row_points[2], row_points[-3]]
            rows.append(row_interval)

            if col_points:
                col_interval = [col_points[2], col_points[-3]]
            cols.append(col_interval)
        return rows, cols

    def _make_canvas_floor(self):
        raw = (self.floor[:, :, 0] - 255) * -1
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        img[:, :, 0] = raw
        img[:, :, 1] = raw
        img[:, :, 2] = raw

        if self.scale > 1:
            img = Image.fromarray(img).resize(size=(self.window_size,) * 2, resample=Image.NEAREST)
        return img

    def _compute_floor_area(self):
        floor = self.floor[:, :, 3]
        floor_area = (floor == 255).sum()
        return floor_area

    def _compute_door_pos(self):
        floor = self.floor[:, :, 0]
        door_positions = list(zip(*np.where(floor == 255)))
        door_pos = np.rint(np.mean(door_positions, axis=0)).astype(np.uint8)  # mean of positions
        return door_pos

    def _distance_to_door(self, room):
        room_pos = self.current_state[room]["pos"]
        euclidean = lambda x, y: np.linalg.norm(x - y)
        return euclidean(room_pos, self.door_pos)

    def reset(self, seed=None, return_info=False):
        # Reset the state of the environment to an initial state
        self.current_state = dict()
        for room in range(self.num_rooms):
            room_dimension = {
                "type": self.rooms_type[room],  # TODO: better in info?
                "size": np.random.randint(1, 5) * 400,
                "proportion": 1,
            }
            self.current_state.update({room: room_dimension})
            self.current_state[room]["pos"] = self._get_initial_position(room)

        obs = self._get_obs()
        return obs

    def _get_initial_position(self, room):
        rnd_pos = np.random.randint(0, 255, size=2)
        area = self.current_state[room]['size']
        prop = self.current_state[room]['proportion']
        b = np.sqrt(area * prop)
        h = (area / b)
        while not self._is_inside_perimeter(rnd_pos, b, h):
            rnd_pos = np.random.randint(0, 255, size=2)
        return rnd_pos

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        reward = self._compute_reward()
        done = self.current_step >= self.steps_per_episode
        info = self._get_info()
        obs = self._get_obs()
        return obs, reward, done, info

    def _is_inside_perimeter(self, pos, b, h):
        x, y = pos
        if not (0 <= x <= 255 and 0 <= y <= 255):
            return False
        if not (self.rows[y] and self.cols[x]):
            return False

        inside_row = self.rows[y][0] + b//2 < x < self.rows[y][1] - b//2
        inside_col = self.cols[x][0] + h//2 < y < self.cols[x][1] - h//2
        return inside_row and inside_col

    def _is_available(self, new_pos, room):
        other_rooms_pos = (state['pos'] for _room, state in self.current_state.items() if _room != room)
        return all(not np.array_equal(new_pos, pos) for pos in other_rooms_pos)

    def _clip_movements(self, prev_pos, movement, room):
        new_pos = prev_pos + movement
        new_pos_is_available = self._is_available(new_pos, room)
        area = self.current_state[room]['size']
        prop = self.current_state[room]['proportion']
        b = np.sqrt(area * prop)
        h = (area / b)
        new_pos_is_inside = self._is_inside_perimeter(new_pos, b, h)
        if new_pos_is_inside and new_pos_is_available:
            return new_pos
        return prev_pos

    def _take_action(self, action):
        # Make changes to our environment
        room, action_id = action

        state_variable, value = self.action_ids[action_id]
        prev_value = self.current_state[room][state_variable]
        if state_variable == "pos":
            self.current_state[room]["pos"] = self._clip_movements(prev_value, value, room)
        else:
            low, high = self.bounds[state_variable]
            clipped_value = np.clip(prev_value + value, low, high)
            # room_is_inside = self._is_inside_perimeter()
            self.current_state[room][state_variable] = clipped_value

    def _compute_reward(self):
        return None

    def _get_obs(self):
        # Return the next observation
        obs = self.current_state
        return obs

    def _get_info(self):
        info = {}
        for room, state in self.current_state.items():
            x, y = state['pos']
            dl, dr = np.abs(self.rows[y] - x)
            dd, dt = np.abs(self.cols[x] - y)
            distances = {
                'left': dl,
                'right': dr,
                'top': dt,
                'down': dd
            }
            info.update({room: {"distances": distances,
                                "dist_to_door": self._distance_to_door(room)
                                }})
        return info

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        if self.window is None and mode == "human":
            # initialize
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        # Main surface (self.window_size, self.window_size) from the floor image
        canvas_floor = pygame.surfarray.make_surface(np.transpose(self.canvas_floor, axes=(1, 0, 2)))

        # Adding each room bounds and centroid
        for room in range(self.num_rooms):
            coord = self.current_state[room]["pos"] * self.scale
            area = (self.current_state[room]["size"]) * self.scale ** 2
            prop = self.current_state[room]["proportion"]
            color = cmap[2][room][0]

            # Now we draw the room
            #   _____________
            #  |             |
            #  |      c      | h
            #  |_____________|
            #         b
            # base (b) and height (h)
            #  area = b*h
            #  prop = b/h
            #  b = srqt(area*prop)

            b = np.sqrt(area * prop)
            h = (area / b)

            # Entire room rectangle
            room_coord = coord + [-b // 2, -h // 2]
            pygame.draw.rect(canvas_floor, color, pygame.Rect(room_coord, (b, h)), width=self.scale)  # Room border

            room_area = pygame.Surface((b, h))  # Tranparency room area
            room_area.set_alpha(100)  # Alpha level determine transparency
            room_area.fill(color)  # this fills the entire surface
            canvas_floor.blit(room_area, room_coord)

            # Room center point
            center_square_size = 2 * self.scale  # The size of the center point in pixels
            center_coord = coord - [center_square_size // 2] * 2
            pygame.draw.rect(canvas_floor, color, pygame.Rect(center_coord, (center_square_size, center_square_size)))

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas_floor, canvas_floor.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
