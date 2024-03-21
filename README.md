
# BlackJackRL

## Overview
BlackJackRL is a new project aimed at exploring and harnessing the power of Deep Reinforcement Learning techniques (RL) to master the game of Blackjack. 

By leveraging various state-of-the-art reinforcement learning algorithms and techniques, this project seeks to not only find the optimal strategy for Blackjack but also to push the boundaries of what's possible in the realm of RL. 

Through continuous experimentation and development, BlackJackRL will also contribute new algorithms to the field, enhancing the understanding and application of reinforcement learning in complex decision-making scenarios.

## Objectives
- To apply and compare different reinforcement learning algorithms in the context of Blackjack.

- To identify the most effective strategies for winning at Blackjack using BlackBox AI techniques.

- To develop new reinforcement learning algorithms based on the insights gained from existing methods.

- Use the weights and strategies learned to put the model to the test in a real online casino.

## Features
- **Flexible Environment**: Simulates the game of Blackjack, allowing for various rule sets and configurations.

- **Algorithm Comparison**: Implements and evaluates multiple reinforcement learning algorithms to find the most effective strategies.

- **Strategy Visualization**: Provides tools for visualizing the strategies learned by the reinforcement learning models.

- **Performance Metrics**: Tracks and displays performance metrics to measure the effectiveness of different strategies and algorithms.

## Getting Started
To get started with BlackJackRL, follow these steps:

1. Clone the repository:
```
git clone https://github.com/AntoniovanDijck/BlackJackRL.git
```
2. Install the required dependencies:
```
cd BlackJackRL
pip install -r requirements.txt
```
3. Run the simulation:
```
python main.py
```

## Algorithms Implemented
- Q-Learning
- Deep Q-Network (DQN)
- MuZero
- Monte Carlo Methods
- (More to be added as the project progresses)

## Project Structure
- `game/blackjack.py`: Defines the Blackjack game environment.
- `classes/algorithms`: Contains implementations of various reinforcement learning agents.
- `weights/`: Stores the neural network models for agents (if applicable).
- `helpers/`: Includes utility functions for data processing and visualization.
- `classes/game`: scripts and classes to run simulations and compare algorithms.

## Todo List
- [ ] Evaluate additional reinforcement learning algorithms.
- [ ] Develop new reinforcement learning algorithms based on project insights.
- [ ] Enhance the simulation environment to include more Blackjack variants.
- [ ] Implement a more sophisticated reward system to better model complex betting strategies.
- [ ] Add support for multi-agent reinforcement learning.
- [ ] Combine reinforcement learning with other machine learning techniques to improve performance.
- [ ] Interface to put the model to the test in a real online casino.


## References
For those interested in diving deeper into reinforcement learning techniques and algorithms, the following resources are highly recommended:

- Sutton, R.S., & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Francois-Lavet, V., Henderson, P., Islam, R., Bellemare, M.G., & Pineau, J. (2018). *An Introduction to Deep Reinforcement Learning*. Foundations and Trends in Machine Learning.
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.
- Lillicrap, T.P., et al. (2016). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.
- Playing Atari with Deep Reinforcement Learning. *arXiv preprint arXiv:1312.5602*.
- Monte Carlo Sampling for regret minimization in extensive games. *arXiv preprint arXiv:1605.06808*.
- Counterfactual Regret Minimization. *arXiv preprint arXiv:1301.5434*.
- LabML: Counterfactual Regret Minimization. https://nn.labml.ai/cfr/
- LabML: Deep Q Networks. https://nn.labml.ai/rl/dqn/
- LabML: Reinforcement Learning. https://nn.labml.ai/rl/ 
- LabML: Proximal Policy Optimization. https://nn.labml.ai/rl/ppo/

This project is still a work in progress, and contributions are welcome. Whether you're interested in developing new algorithms, enhancing the simulation environment, or analyzing the strategies learned by the models, your input can help shape the future of BlackJackRL.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The developers and contributors of the open-source reinforcement learning libraries used in this project.
- The creators of the original Blackjack game, which has provided the inspiration for this project.
- The broader reinforcement learning community, whose work continues to push the boundaries of what's possible in the field.

## Authors
- Antonio van Dyck


