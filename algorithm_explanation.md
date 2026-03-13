# Linear Q-Learning with Traffic Avoidance

## Overview

A reinforcement learning agent navigates an 8x8 grid from start (0,0) to target (7,7), learning to avoid high-traffic cells and generalize to unseen traffic patterns.

---

## 1. Environment

- **Grid**: 8x8 cells
- **Start**: (0, 0) — top-left
- **Target**: (7, 7) — bottom-right
- **Traffic**: Random cells assigned traffic values between 0.6 and 1.0
- **Actions**: 4 discrete moves — right, up, left, down

### Action-to-Direction Mapping

| Action | Direction | Vector  |
|--------|-----------|---------|
| 0      | Right     | (0, +1) |
| 1      | Up        | (-1, 0) |
| 2      | Left      | (0, -1) |
| 3      | Down      | (+1, 0) |

### Movement

The agent's new position after taking action $a$ is:

$$
(x', y') = \text{clip}((x, y) + \text{direction}(a), \; 0, \; 7)
$$

Clipping keeps the agent inside the grid.

---

## 2. Reward Function

At each step, the agent receives a reward based on the cell it lands on:

$$
R(s, a) =
\begin{cases}
+10 & \text{if agent reaches target (7,7)} \\
-5 & \text{if traffic at cell} > 0.5 \\
-0.1 & \text{otherwise (step cost)}
\end{cases}
$$

- **+10** gives a strong signal for reaching the goal
- **-5** penalizes entering high-traffic cells
- **-0.1** encourages shorter paths (every step has a small cost)

---

## 3. State Representation

The agent observes:

1. Its current position $(x, y)$
2. The traffic level in each of the 4 adjacent cells: $(t_R, t_U, t_L, t_D)$

Where $t_R$ is the raw traffic value (0.0 to 1.0) of the cell to the right, etc. If the adjacent cell is outside the grid, the traffic value is 0.

---

## 4. Feature Vector (Per Action)

Instead of a lookup table, we compute **features for each candidate action** $a$ from state $s = (x, y, t_R, t_U, t_L, t_D)$:

$$
\phi(s, a) = \begin{bmatrix}
\text{moves\_closer}(s, a) \\
\text{traffic}(s, a) \\
\text{traffic\_blocks\_path}(s, a) \\
\text{hits\_wall}(s, a) \\
1
\end{bmatrix}
$$

### Feature Definitions

| Feature | Definition | Value |
|---------|-----------|-------|
| `moves_closer` | Does action $a$ reduce Manhattan distance to target? | 1.0 if yes, 0.0 if no |
| `traffic` | Traffic level in the direction of action $a$ | $t_a \in [0.0, 1.0]$ |
| `traffic_blocks_path` | Interaction: traffic on the direct route | `moves_closer` $\times$ `traffic` |
| `hits_wall` | Would action $a$ result in no movement (wall)? | 1.0 if yes, 0.0 if no |
| bias | Constant | 1.0 |

### Manhattan Distance

$$
d(x, y) = |x - 7| + |y - 7|
$$

### moves_closer

$$
\text{moves\_closer}(s, a) =
\begin{cases}
1 & \text{if } d(x', y') < d(x, y) \\
0 & \text{otherwise}
\end{cases}
$$

### traffic_blocks_path (Interaction Term)

$$
\text{traffic\_blocks\_path}(s, a) = \text{moves\_closer}(s, a) \times \text{traffic}(s, a)
$$

This is the key feature. It captures: "I want to go this direction (toward target), BUT there's traffic blocking it." Without this term, the agent can't learn to reroute.

---

## 5. Q-Value Computation

A single shared weight vector $\mathbf{w} \in \mathbb{R}^5$ is used for all actions:

$$
Q(s, a) = \mathbf{w}^\top \phi(s, a)
$$

To evaluate all 4 actions from state $s$:

$$
Q(s, a_i) = \mathbf{w}^\top \phi(s, a_i) \quad \text{for } i = 0, 1, 2, 3
$$

The agent picks the action with the highest Q-value (during exploitation).

---

## 6. Action Selection (Epsilon-Greedy)

$$
a =
\begin{cases}
\text{random action from } \{0,1,2,3\} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}
$$

### Epsilon Decay

After each episode:

$$
\epsilon \leftarrow \max(\epsilon_{\min}, \; \epsilon \times \lambda)
$$

Where:
- $\epsilon_{\min} = 0.05$
- $\lambda = 0.9997$ (decay rate)

This starts with full exploration ($\epsilon = 1.0$) and gradually shifts to exploitation.

---

## 7. Q-Learning Update (with Linear Function Approximation)

After taking action $a$ in state $s$, observing reward $R$ and arriving in state $s'$:

### Step 7a: Compute TD Target

$$
\text{target} =
\begin{cases}
R & \text{if } s' \text{ is terminal (reached target)} \\
R + \gamma \cdot \max_{a'} Q(s', a') & \text{otherwise}
\end{cases}
$$

Where $\gamma = 0.95$ is the discount factor.

### Step 7b: Compute TD Error

$$
\delta = \text{target} - Q(s, a)
$$

### Step 7c: Update Weights

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \delta \cdot \phi(s, a)
$$

Where $\alpha = 0.005$ is the learning rate.

This is **semi-gradient Q-learning** — standard Q-learning but with a linear function approximation instead of a table.

---

## 8. Training Algorithm (Step by Step)

```
Initialize weights w = [0, 0, 0, 0, 0]
Initialize epsilon = 1.0

For each episode (1 to 20,000):
    1. Generate random traffic layout (6 random high-traffic cells)
    2. Place agent at (0, 0)

    For each step (1 to 200):
        3. Observe state s = (x, y, traffic in 4 directions)
        4. Compute features phi(s, a) for all 4 actions
        5. Compute Q(s, a) = w dot phi(s, a) for all 4 actions
        6. Select action a using epsilon-greedy
        7. Execute action a, observe reward R and new state s'
        8. Compute TD target:
             If s' is terminal: target = R
             Else: target = R + 0.95 * max Q(s', a')
        9. Compute TD error: delta = target - Q(s, a)
       10. Update weights: w = w + 0.005 * delta * phi(s, a)
       11. If terminal, end episode

    Decay epsilon: epsilon = max(0.05, epsilon * 0.9997)
```

---

## 9. Testing (Generalization)

```
Set epsilon = 0 (fully greedy, no exploration)
Generate NEW traffic layout (never seen during training)
Place agent at (0, 0)

For each step:
    1. Observe state s
    2. Compute Q(s, a) for all actions
    3. Pick a = argmax Q(s, a)
    4. Execute action a
    5. If reached target, done
```

The agent generalizes because the weights encode **general rules**, not memorized positions:

| Weight | Meaning |
|--------|---------|
| $w_0$ (moves_closer) | Positive → prefer moving toward target |
| $w_1$ (traffic) | Negative → avoid high-traffic directions |
| $w_2$ (traffic_blocks_path) | Negative → especially reroute when traffic is on the direct path |
| $w_3$ (hits_wall) | Negative → don't walk into walls |
| $w_4$ (bias) | Baseline action value |

---

## 10. Hyperparameters Summary

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Learning rate | $\alpha$ | 0.005 |
| Discount factor | $\gamma$ | 0.95 |
| Initial exploration | $\epsilon_0$ | 1.0 |
| Minimum exploration | $\epsilon_{\min}$ | 0.05 |
| Exploration decay | $\lambda$ | 0.9997 |
| Episodes | — | 20,000 |
| Max steps per episode | — | 200 |
| Grid size | — | 8 x 8 |
| Number of features | — | 5 |
| Number of weights | — | 5 |
