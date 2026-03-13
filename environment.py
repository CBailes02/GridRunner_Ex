# Adding reversal learning

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RL_Environment:
    def __init__(self):        
        # Q_sa = [[0], [0], [0], [0]]
        # Q_sb = [[0], [0], [0], [0]]
        self.p_a = 0.7
        self.p_b = 0.3
        self.action_list = [1, 0]
        self.action_a = 1
        self.action_b = 0
        self.action_dict = {1: "A", 0: "B"}
        self.accurate_count_list = []
        self.max_steps = 1000
        self.switch_prob = 0.01
        self.state = 1 if random.random() < self.p_a else 0    #70% of time state is = 1 initially
        self.action = self.action_a
        self.reward = 0
        self.outcome = 0
        
    def step(self, action, step):
        if self.state == 1:
            self.outcome = 1 if random.random() < self.p_a else 0    #70% of time outcome = 1, independent of action
        else:
            self.outcome = 1 if random.random() < self.p_b else 0    #30% of time outcome = 1, ind. of action

        self.reward = 1 if action == self.outcome else 0

        if random.random() < self.switch_prob:
            self.state = 1 - self.state
        
        # if step == 500:
        #     self.state = 1 - self.state

        return self.state, self.reward, self.outcome
        

class RL_Bandit_Agent:
    def __init__(self):
        self.alpha = 0.2
        self.B = 5.0
        # self.Q_sa = np.zeros((2, 2))
        self.probs_a = 1
        self.probs_b = 0
        self.action_a = 1
        self.action_b = 0
        self.running_accuracy = None
        self.Q_sa = np.zeros((2,))
        self.accurate_count_list = []
        self.action = random.choice([0,1])
        #self.reward = 0

    def take_action(self, reward):
        #self.reward = reward
        self.probs_a = (math.exp(self.B * self.Q_sa[self.action_a])) / float(
            np.sum(
                (
                    math.exp(self.B * self.Q_sa[self.action_a]),
                    math.exp(self.B * self.Q_sa[self.action_b]),
                )
            )
        )

        # probs_b = (math.exp(B*Q_sb[state, action])) / float(np.sum((math.exp(B*Q_sa[state, action]), math.exp(B*Q_sb[state, action]))))
        self.probs_b = 1 - self.probs_a

        self.action = self.action_a if random.random() < self.probs_a else self.action_b
        #reward = 1 if action == outcome else 0
        #self.Q_sa[self.action] = self.Q_sa[self.action] + self.alpha * (reward - self.Q_sa[self.action])

        self.accurate_count_list.append(reward)

        self.running_accuracy = np.cumsum(self.accurate_count_list) / np.arange(
            1, len(self.accurate_count_list) + 1
        )

        return self.action
    
    def update_Q(self, reward):
        self.Q_sa[self.action] = self.Q_sa[self.action] + self.alpha * (reward - self.Q_sa[self.action])
        #return self.Q_sa

    def plot_accuracy(self):
        plt.figure(figsize=(8, 6))
        plt.title("Running Action Accuracy")
        plt.plot(self.running_accuracy)
        plt.show()
        


env =  RL_Environment()
agent = RL_Bandit_Agent()
reward = env.reward
state = env.state
outcome = env.outcome
action = env.action

print("state: ", state)


for step in range(0, env.max_steps):
    action = agent.take_action(reward)
    state, reward, outcome = env.step(action, step)
    agent.update_Q(reward)

    #action = agent.take_action(reward, outcome)

    print(step, state, action, outcome)

plt.figure(figsize=(8, 6))
plt.title("Running Action Accuracy")
plt.plot(agent.running_accuracy)
plt.show()


agent.plot_accuracy()





# # Q_sa = [[0], [0], [0], [0]]
# # Q_sb = [[0], [0], [0], [0]]
# Q_sa = np.zeros((2, 2))
# alpha = 0.2
# p_a = 0.7
# p_b = 0.3
# B = 5.0
# action_list = [1, 0]
# action_a = 1
# action_b = 0
# action_dict = {1: "A", 0: "B"}
# accurate_count_list = []

# state = 1 if random.random() < p_a else 0    #70% of time state is = 1 initially
# action = action_a
# Q00_hist = []
# Q01_hist = []
# Q10_hist = []
# Q11_hist = []
# Q_sa = np.zeros((2, ))
# for step in range(0, max_steps):
#     probs_a = (math.exp(B*Q_sa[action_a])) / float(np.sum((math.exp(B*Q_sa[action_a]), math.exp(B*Q_sa[action_b]))))
# #    probs_b = (math.exp(B*Q_sb[state, action])) / float(np.sum((math.exp(B*Q_sa[state, action]), math.exp(B*Q_sb[state, action]))))
#     probs_b = 1 - probs_a
#     action = action_a if random.random() < probs_a else action_b

#     if state == 1:
#         outcome = 1 if random.random() < p_a else 0    #70% of time outcome = 1
#     else:
#         outcome = 1 if random.random() < p_b else 0    #30% of time outcome = 1
    
#     reward = 1 if action == outcome else 0

#     Q_sa[action] = Q_sa[action] + alpha*(reward-Q_sa[action])
#     #state = 1 if random.random() < p_a else 0

#     #state = 1 if outcome == action  else 0

#     # Q00_hist.append(Q_sa[0,0])
#     # Q01_hist.append(Q_sa[0,1])
#     # Q10_hist.append(Q_sa[1,0])
#     # Q11_hist.append(Q_sa[1,1])
#     #Q_sa[state, action] = Q_sa[state, action] + alpha*(reward-Q_sa[state, action])

#     # elif action == action_b:
#     #     Q_sb[state, action] = Q_sb[state, action] + alpha*(reward-Q_sb[state, action])

#     print(step, state, action, outcome)

#     # print("Q[a] Value: ",round(Q_sa[state, 1], 3), "  Q[b] Value: ", round(Q_sa[state, 0],3), "Action Taken: ", action_dict[action], 
#     #       "Outcome: ", action_dict[outcome])

#     acc_check = 1 if action == outcome else 0
#     accurate_count_list.append(acc_check)
#     running_acc = np.cumsum(accurate_count_list) / np.arange(1, len(accurate_count_list)+1)


#     if random.random() < 0.50:
#         state = 1 - state

# plt.figure(figsize=(8, 6))
# plt.title("Running Action Accuracy")
# plt.plot(running_acc)


# plt.figure(figsize=(8,6))

# plt.plot(Q00_hist, label="Q(state0,B)")
# plt.plot(Q01_hist, label="Q(state0,A)")
# plt.plot(Q10_hist, label="Q(state1,B)")
# plt.plot(Q11_hist, label="Q(state1,A)")

# plt.xlabel("Step")
# plt.ylabel("Q value")
# plt.title("Q-table values over learning steps")

# plt.legend()
# plt.show()

# plt.show()
# avg_Q = total/1000
# print(avg_Q)