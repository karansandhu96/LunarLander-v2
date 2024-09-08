
## Actor-Critic Reinforcement Learning Model for LunarLander-v2

This project implements an Actor-Critic Reinforcement Learning model to solve the LunarLander-v2 environment from OpenAI's Gym. The notebook demonstrates the implementation of the Actor and Critic networks, training the model, and evaluating its performance through cumulative rewards.

### Project Overview:
1. **Environment Setup:**
   - **LunarLander-v2:** The environment is set up using the `gym` library, where the goal is to land the lunar module safely on the landing pad.
   - **Rendering:** The environment is rendered every 10 episodes during training and in every episode during evaluation to visualize the agent's performance.

2. **Model Architecture:**
   - **Actor Network:** 
     - The Actor network selects actions based on the current state by outputting a probability distribution over actions. It consists of three fully connected layers with ReLU activations.
   - **Critic Network:** 
     - The Critic network estimates the value of the current state, providing feedback to the Actor network. It also consists of three fully connected layers with ReLU activations.
   - **Optimizers:**
     - Adam optimizers are used for both Actor and Critic networks to minimize their respective loss functions.

3. **Training Process:**
   - The model is trained over a series of episodes, where the Actor network learns to take actions that maximize cumulative rewards, while the Critic network learns to evaluate the actions taken by the Actor.
   - The training loop updates both the Actor and Critic networks using the rewards obtained from the environment.

4. **Evaluation:**
   - **Trained Model Evaluation:**
     - After training, the model is evaluated over a set of episodes to measure its performance.
   - **Random Policy Evaluation:**
     - A random policy is also evaluated to provide a baseline comparison against the trained model.

5. **Plotting the Learning Curve:**
   - The learning curve is plotted to show the cumulative rewards obtained by the trained Actor-Critic model, the evaluation phase, and the random policy.

### How to Use:
1. **Clone the Repository:**
   - Clone the repository to your local machine using `git clone`.
   
2. **Install Dependencies:**
   - Install the required Python libraries listed in the `requirements.txt` file using `pip install -r requirements.txt`.

3. **Run the Notebook:**
   - Open the notebook in Jupyter and execute the cells sequentially to train the model, evaluate its performance, and generate the results.

### Visualizations:
This project includes a key visualization to help interpret the model's performance:

<img width="364" alt="image" src="https://github.com/user-attachments/assets/e6fe937e-b731-4e1e-a2f6-85a0f36dcea6">


1. **Learning Curve:**
   
  <img width="729" alt="image" src="https://github.com/user-attachments/assets/6c4c7ab0-4273-4dda-a2f2-cefae0505606">

### Conclusion:
This project demonstrates the implementation of an Actor-Critic reinforcement learning model for the LunarLander-v2 environment. The model is trained to achieve safe landings, and its performance is evaluated and compared with a random policy baseline. The learning curve visualization highlights the effectiveness of the Actor-Critic approach in this context.
