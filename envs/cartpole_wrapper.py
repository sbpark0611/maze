import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import numpy as np
from math import inf
from PIL import Image
import copy
import torch
import torchvision.transforms as T


class ImageWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        subspace1 = spaces.Box(low=0, high=1, shape=(64, 144, 6), dtype=np.float32)
        subspace2 = spaces.Discrete(2)
        subspace3 = spaces.Box(low=0, high=1, shape=(64, 144, 6), dtype=np.float32)

        self.env.observation_space = spaces.Dict({
            'image': subspace1,
            'prev_action': subspace2,
            'prev_image': subspace3,
        })

        self.max_episode_steps = 500
    
    def reset(self, seed = 0):
        obs, i = self.env.reset()
        self.prev_image = np.array(self.get_screen()).transpose((1, 2, 0))
        self.prev_prev_image = copy.deepcopy(self.prev_image)

        self.steps = 0

        new_obs = {"image": np.concatenate((self.prev_image, self.prev_image), axis=2), "prev_action": 0, "prev_image": np.concatenate((self.prev_image, self.prev_prev_image), axis=2)}


        return new_obs, {"steps": self.steps}

    def step(self, action):
        obs, reward, done, t, info = self.env.step(action)
        image = np.array(self.get_screen()).transpose((1, 2, 0))

        new_obs = {"image": np.concatenate((image, self.prev_image), axis=2), "prev_action": action, "prev_image": np.concatenate((self.prev_image, self.prev_prev_image), axis=2)}
        self.prev_prev_image = self.prev_image
        self.prev_image = image
        
        self.steps += 1

        return new_obs, reward, done, t, {"steps": self.steps}

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  
    
    def get_screen(self):
        FRAMES = 2
        RESIZE_PIXELS = 64

        resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZE_PIXELS, Image.NEAREST),
                    #T.Grayscale(),
                    T.ToTensor()])

        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render().transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return resize(screen)


def vision_cart_pole():
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped 
    env = ImageWrapper(env)
    return env


if __name__=="__main__":
    env = vision_cart_pole()
    observation, i = env.reset(0)
    done = False
    while not done:
        action = int(input("action: "))
        observation, reward, done, t, info = env.step(action)