import torch
import torch.nn as nn

import numpy as np
from PIL import Image

import cv2

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
        self.goal_embedding = nn.Linear(3, embedding_size)
        self.planner = Planner(embedding_size, num_heads, num_iterations)
        self.planner_mlp = nn.Sequential(
            nn.Linear(
                self.planner.embedding_size + embedding_size, self.planner.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.planner.hidden_size, self.planner.embedding_size),
        )

        self.feature = nn.Sequential(
            nn.Linear(embedding_size * 2 + self.planner.embedding_size, hidden_size),
            nn.ReLU(),
        )

        self.in_channels = 3
        kernels = (3, 3, 3)
        stride = 2
        d = 4
        self.image_embedding_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, d, kernels[0], stride),
            nn.ELU(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            nn.ELU(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            nn.Flatten()
        )
        self.image_embedding_linear = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, embedding_size)
        )

    def visualization(self, image, step, reward):
        # 텍스트 정보 설정
        text1 = "Steps: " + str(step)
        text2 = "Rewards: " + str(reward)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (255, 255, 255)
        thickness = 5
        line_type = cv2.LINE_AA

        # 텍스트 크기 계산
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]

        margin = 50

        # 텍스트를 표시할 이미지의 크기 계산
        text1_width, text1_height = text1_size[0], text1_size[1]
        text2_width, text2_height = text2_size[0], text2_size[1]
        image_height, image_width = image.shape[0], image.shape[1]
        combined_image = np.zeros((image_height, image_width + max(text1_width, text2_width) + margin * 2, 3), dtype=np.uint8)

        # 이미지 복사
        combined_image[:, :image_width] = image

        # 텍스트를 합쳐진 이미지의 오른쪽에 추가
        text_x = image_width + margin  # 텍스트와 이미지 사이의 간격
        text1_y = text1_height // 2 + (margin * 2)
        text2_y = text1_height + text2_height // 2 + (margin * 3)
        cv2.putText(combined_image, text1, (text_x, text1_y), font, font_scale, font_color, thickness, line_type)
        cv2.putText(combined_image, text2, (text_x, text2_y), font, font_scale, font_color, thickness, line_type)

        # 결과 이미지 표시
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(1)

    def forward(self, memory, obs):
        # print(obs['image'][0]*100)
        # self.visualization(cv2.resize(np.array(obs["image"][0]*100), (720, 720), interpolation = cv2.INTER_AREA), 0, 0)

        memory_image = memory["image"].to(torch.float32).permute(0, 1, 4, 2, 3)
        memory_prev_image = memory["prev_image"].to(torch.float32).permute(0, 1, 4, 2, 3)
        obs_image = obs["image"].to(torch.float32).permute(0, 3, 1, 2)
        
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

        goal = self.goal_embedding(obs["goal"].to(torch.float32))
        goals = goal.unsqueeze(1).expand(-1, fixed_num_steps, -1)

        episodic_storage = torch.concat((states, actions, prev_states, goals), dim=-1)

        belief_state = self.planner(episodic_storage)

        current_state = self.image_embedding_conv(obs_image)
        current_state = current_state.reshape(current_state.shape[0], -1)
        current_state = self.image_embedding_linear(current_state)

        current_states = current_state.unsqueeze(1).expand(-1, fixed_num_steps, -1)
        belief_state = torch.cat((belief_state, current_states), dim=2)
        belief_state = self.planner_mlp(belief_state)
        planner_output = torch.max(belief_state, dim=1)[0]

        state_goal_embedding = torch.concat((current_state, goal), dim=-1)
        combined_embedding = torch.cat((planner_output, state_goal_embedding), dim=1)
        combined_embedding = self.feature(combined_embedding)
        return combined_embedding


class Planner(nn.Module):
    def __init__(self, embedding_size, num_heads, num_iterations, dropout=0):
        super(Planner, self).__init__()
        self.embedding_size = 4 * embedding_size
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
