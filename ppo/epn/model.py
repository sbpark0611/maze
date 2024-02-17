import torch
import torch.nn as nn

import numpy as np
from PIL import Image


class EPN(nn.Module):
    def __init__(
        self,
        num_actions: int,
        embedding_size: int = -1,
        num_heads: int = -1,
        hidden_size: int = -1,
        num_iterations: int = -1,
    ):
        super(EPN, self).__init__()
        self.hidden_size = hidden_size
        self.action_embedding = nn.Embedding(num_actions + 1, embedding_size)
        self.planner = Planner(embedding_size, num_heads, num_iterations)
        self.planner_mlp = nn.Sequential(
            nn.Linear(
                self.planner.embedding_size + embedding_size, self.planner.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.planner.hidden_size, self.planner.embedding_size),
        )

        self.feature = nn.Sequential(
            nn.Linear(embedding_size + self.planner.embedding_size, hidden_size),
            nn.ReLU(),
        )

        self.in_channels = 3
        kernels = (3, 3, 3)
        stride = 1
        d = 4
        self.image_embedding_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, d, kernels[0], stride),
            nn.ELU(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            nn.ELU(),
            nn.Flatten()
        )
        self.image_embedding_linear = nn.Sequential(
            nn.Linear(2704, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, embedding_size)
        )

        self.n = 0


    def custom_imshow(self, img, name):
        import matplotlib.pyplot as plt

        img = Image.fromarray(img)
        img.show()
        #img.save(name, 'JPEG')

    def isi(self, a):
        return a != 1

    def forward(self, memory, obs):
        memory_image = memory["image"].to(torch.float32).permute(0, 1, 4, 2, 3)
        memory_prev_image = memory["prev_image"].to(torch.float32).permute(0, 1, 4, 2, 3)
        obs_image = obs["image"].to(torch.float32).permute(0, 3, 1, 2)
        
        #self.custom_imshow(obs_image.permute(0,2,3,1).numpy().astype(np.uint8)[0], str(self.n))
        self.n += 1

        states = []
        for i in range(memory_image.size()[0]):
            new_states = self.image_embedding_conv(memory_image[i])
            new_states = new_states.view(new_states.shape[0], -1)
            new_states = self.image_embedding_linear(new_states)
            states.append(new_states)
        states = torch.stack(states, dim=0)
        _, fixed_num_steps, _ = states.size()

        actions = self.action_embedding(memory["prev_action"])

        prev_states = []
        for i in range(memory_prev_image.size()[0]):
            new_prev_states = self.image_embedding_conv(memory_prev_image[i])
            new_prev_states = new_prev_states.view(new_prev_states.shape[0], -1)
            new_prev_states = self.image_embedding_linear(new_prev_states)
            prev_states.append(new_prev_states)
        prev_states = torch.stack(prev_states, dim=0)

        episodic_storage = torch.concat((states, actions, prev_states), dim=-1)

        belief_state = self.planner(episodic_storage)

        current_state = self.image_embedding_conv(obs_image)
        current_state = current_state.reshape(current_state.shape[0], -1)
        current_state = self.image_embedding_linear(current_state)

        current_states = current_state.unsqueeze(1).expand(-1, fixed_num_steps, -1)
        belief_state = torch.cat((belief_state, current_states), dim=2)
        belief_state = self.planner_mlp(belief_state)
        planner_output = torch.max(belief_state, dim=1)[0]

        state_goal_embedding = current_state
        combined_embedding = torch.cat((planner_output, state_goal_embedding), dim=1)
        combined_embedding = self.feature(combined_embedding)
        return combined_embedding


class Planner(nn.Module):
    def __init__(self, embedding_size, num_heads, num_iterations, dropout=0):
        super(Planner, self).__init__()
        self.embedding_size = embedding_size * 3
        self.num_heads = num_heads
        self.hidden_size = 16 * embedding_size
        self.num_iterations = num_iterations
        self.ln_1 = nn.LayerNorm(self.embedding_size)
        self.self_attn = nn.MultiheadAttention(
            self.embedding_size, num_heads, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(self.embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.embedding_size),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(self.embedding_size)

    def forward(self, x):
        for _ in range(self.num_iterations):
            _x = self.ln_1(x)
            attn_output, _ = self.self_attn(_x, _x, _x)
            x = x + self.dropout1(attn_output)
            x = x + self.dropout2(self.mlp(self.ln_2(x)))
        return self.ln_f(x)
