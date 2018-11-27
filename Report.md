# Project 3 : Collaboration and Competition

## Report

## Learning Algorithm

In this project, we used the Deep Deterministic Policy Gradients ([DDPG](https://arxiv.org/abs/1509.02971)) learning algorithm to solve the Tennis environment. DDPG is an off-policy model-free algorithm that uses neural networks to learn policies, even in high-dimensional and continuous action spaces. The implementation has two DDPG agents with shared actor and critic networks. Each agent uses the same actor network to take an action, sampled from a shared replay buffer. Both actor and critic have three fully connected layers: hidden layers of 500 and then 375 units, each with ReLU activation, and an output layer (of 2 units for the actor, with tanh applied in order to bound the output between -1 and 1, and of 1 unit for the critic). In addition, in the critic network, the action vector is included between the first and second hidden layers. The hyperparameters are as follows:

| Hyperparameter | Value |
| ------------- | ------------- |
| first hidden layer units | 500 |
| second hidden layer units | 375 |
| replay buffer size | 1e5 |
| batch size | 128 |
| discount factor (gamma) | 0.99 |
| tau* | 3e-1 |
| actor learning rate | 1e-4 |
| critic learning rate | 1e-4 |
| number of episodes | 5000 |
| L2 weight decay | 0 |

*Tau is the percentage of weights from the local model to carry over to the target model during the soft update of target parameters; meanwhile, `1 - tau` is the percentage of target model weights to carry over.

## Results

| Trial | # of Episodes to Solve | Description | Comments |
| ------------- | ------------- | ------------- | ------------- |
| Initial Run | 518 | Default params with tau 1e-2, batch size 128, and critic LR 1e-3 | Baseline [Reacher] code with a change to tau |
| Trial 1 | Interrupted after 3500 | Tau 1e-3 | Not solved |
| Trial 2 | 820 | Tau 1e-1 | Better than Initial Run |
| Trial 3 | Interrupted after 1900 | Tau 5e-1 | Close to solving but not better than Initial Run and Trial 2 |
| Trial 4 | 534 | Tau 3e-1 | Best results thus far |
| Trial 5 | Interrupted after 1700 | Tau 4e-1 | Not that close to solving |
| Trial 6 | 601 | Tau 2e-1 | Not better than Trial 4 |
| Trial 7 | 702 | Tau 3e-1 and batch size 256 | Not better than Trial 4 |
| Trial 8 | 564 | Tau 3e-1 and batch size 512 | Not better than Trial 4 |
| Trial 9 | 571 | Tau 3e-1 and batch size 1024 | Not better than Trial 4 |
| Trial 10 | Interrupted after 1000 | Tau 3e-1 and batch size 64 | Score did not reach 0.2 |
| Trial 11 | Interrupted after 1000 episodes | Tau 3e-1 and batch size 128; actor LR 1e-3 | Plateaued at score 0.0 after episode 100 |
| Trial 12 | Interrupted after 1800 episodes | Tau 3e-1 and batch size 128; actor LR 1e-2 and critic LR 1e-2 | Plateaued at score 0.0 after episode 100 |
| Trial 13 | 457 | Tau 3e-1 and batch size 128; critic LR 1e-4 | Best results thus far, better than Trial 4 |
| Trial 14 | Interrupted after 600 episodes | Tau 3e-1 and batch size 128; actor LR 5e-5 and critic LR 5e-5 | Not better than Trial 13 |
| Final Run | 457 | Tau 3e-1 and batch size 128; critic LR 1e-4 | Verifying Trial 13 performs best |

## Plot of Rewards

The plot below shows that, after 457 episodes, the agent is able to receive an average reward of 0.5 over the last 100 consecutive episodes.

![final_model_rewards_plot](./final_model_rewards_plot.png)

## Ideas for Future Work

Ideas for improving the agent's performance are as follows:
- Attempt prioritized experience replay and D4PG.
- Add lots of noise at the beginning and then reduce or remove it completely after a certain number of episodes.
- Update the weights multiple times per timestep.
- Implement MADDPG: separate actors, separate centralized critics, and a shared replay buffer.
- Try a variation with: separate actors, one shared centralized critic, and a shared replay buffer.
