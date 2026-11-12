# 4. Actor-Critic 与 GAE

第 3 章以下式作结：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) A^\pi(s_t, a_t) \right]
$$

同时留下了一个悬而未决的问题：我们并不知道 $A^\pi$。本章讲的就是如何估计它，以及这样一个事实：每一种估计量都位于偏差–方差谱上的某个位置，而这个位置由你来选。

## 4.1 actor 与 critic 的分工

两个网络，两份工作。

- **actor** 是 $\pi_\theta(a \mid s)$。它负责选择动作。
- **critic** 是 $\hat V_\psi(s) \approx V^\pi(s)$。它负责给状态打分。

critic 从不做任何选择。它唯一的用途就是提供基线，把高方差的回报变成低方差的优势。它通过回归来训练：

$$
L_V(\psi) = \mathbb{E}\left[ \big( \hat V_\psi(s_t) - \hat G_t \big)^2 \right]
\tag{4.1}
$$

其中 $\hat G_t$ 是回报的某个目标值。到底该用哪个目标，正是本章要回答的问题。

在 `ppo_from_scratch/ppo.py` 中，这两者分别是 `ActorCritic.actor` 和 `ActorCritic.critic`——两个独立的 MLP，没有共享主干。原因写在类的 docstring 里，也写在第 6 章中。

## 4.2 TD 误差

整套方法里最重要的那个量：

$$
\delta_t = r_t + \gamma \hat V(s_{t+1}) - \hat V(s_t)
\tag{4.2}
$$

*实际发生的，减去我预测会发生的。*

如果 critic 是精确的，贝尔曼方程（2.6）告诉我们 $\mathbb{E}[\delta_t] = 0$：预测与现实在平均意义上一致。非零的 $\delta_t$ 就是意外，而意外就是信息。

关键事实，也是 $\delta$ 无处不在的原因：**如果 $\hat V = V^\pi$，那么 $\delta_t$ 就是 $A^\pi(s_t, a_t)$ 的无偏估计。**

证明。按定义，$Q^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})]$。于是

$$
\mathbb{E}[\delta_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})] - V^\pi(s_t)
= Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t,a_t)
$$

这就是单步 actor-critic 的全部依据。

## 4.3 $n$ 步回报：一条谱系

$\delta_t$ 只用一个真实奖励，然后就信任 critic。蒙特卡洛回报用上所有真实奖励，从不信任 critic。两者之间是一整族估计：

$$
\begin{aligned}
\hat A_t^{(1)} &= r_t + \gamma \hat V(s_{t+1}) - \hat V(s_t) = \delta_t \\
\hat A_t^{(2)} &= r_t + \gamma r_{t+1} + \gamma^2 \hat V(s_{t+2}) - \hat V(s_t) = \delta_t + \gamma \delta_{t+1} \\
\hat A_t^{(3)} &= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} \\
&\quad \quad \vdots \\
\hat A_t^{(n)} &= \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
\end{aligned}
\tag{4.3}
$$

式 4.3 中的伸缩相消值得亲手验证一次。对 $n = 2$：

$$
\delta_t + \gamma\delta_{t+1}
= \big[r_t + \gamma \hat V_{t+1} - \hat V_t\big] + \gamma\big[r_{t+1} + \gamma \hat V_{t+2} - \hat V_{t+1}\big]
= r_t + \gamma r_{t+1} + \gamma^2 \hat V_{t+2} - \hat V_t
$$

中间的 $\hat V_{t+1}$ 被消掉了。每一个 $n$ 步优势都是连续若干个 TD 误差的折扣和。

沿着这一族估计的权衡：

| | 小 $n$ | 大 $n$ |
|---|---|---|
| 依赖于 | critic | 实际奖励 |
| **偏差** | 高——继承 $\hat V$ 中的每一处误差 | 低——真实回报，不依赖模型 |
| **方差** | 低——只有一个随机奖励 | 高——$n$ 个随机奖励之和 |

两端都不对。$n=1$ 配上一个错了 20% 的 critic，给出的梯度就会系统性地错 20%。$n=\infty$ 给你一个无偏的梯度，但噪声大到你根本迈不出有用的一步。

## 4.4 广义优势估计

GAE（Schulman et al., 2015）拒绝挑选某一个 $n$，而是对所有这些估计取指数加权平均：

$$
\hat A_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat A_t^{(n)}
\tag{4.4}
$$

$(1-\lambda)$ 用于归一化权重，因为 $\sum_{n\ge1}\lambda^{n-1} = 1/(1-\lambda)$。

### 推导闭式解

把式 4.3 代入式 4.4，并交换求和次序：

$$
\begin{aligned}
\hat A_t
&= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \sum_{l=0}^{n-1} \gamma^l \delta_{t+l} \\
&= (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \sum_{n=l+1}^{\infty} \lambda^{n-1}
&& \text{(each } \delta_{t+l} \text{ appears in every } n > l\text{)} \\
&= (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \cdot \frac{\lambda^{l}}{1-\lambda}
&& \textstyle \left(\sum_{n=l+1}^{\infty}\lambda^{n-1} = \lambda^l/(1-\lambda)\right) \\
&= \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
\end{aligned}
$$

$$
\boxed{\quad \hat A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}\quad }
\tag{4.5}
$$

一个对 TD 误差的单一指数折扣和。所有这些机制最终坍缩成一个只需一次反向遍历就能算出来的东西：

$$
\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}
\tag{4.6}
$$

加上回合边界掩码之后，它恰好就是 `RolloutBuffer.compute_returns` 中的那个循环：

```python
for t in reversed(range(self.n_steps)):
    next_non_terminal = 1.0 - self.dones[t + 1]
    next_value = self.values[t + 1]
    delta = rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    advantages[t] = last_gae
```

正是 `next_non_terminal` 这个因子让式 4.6 尊重回合边界：在真正的终止处，未来没有任何贡献，所以自举项和迹都被清零。

### 带边界的完整形式

$$
\hat A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \left( \prod_{j=1}^{l} (1 - d_{t+j}) \right) \delta_{t+l}
\tag{4.7}
$$

其中 $d$ 是终止指示变量。实际实现的正是式 4.7。

## 4.5 $\lambda$ 的作用

两个特殊情形分别恢复出两个端点：

- $\lambda = 0$：$\hat A_t = \delta_t$。纯单步 TD。方差最小，对 critic 的依赖最大。
- $\lambda = 1$：$\hat A_t = \sum_l \gamma^l \delta_{t+l} = G_t - \hat V(s_t)$（这个和完全伸缩相消）。即蒙特卡洛。无偏，方差最大。

所以 $\lambda$ 是一个**偏差–方差旋钮**，而且它是真正连续的：0.95 的意思是“在接下来 $\sim 1/(1-\gamma\lambda) \approx 17$ 步内主要信任真实奖励，然后把接力棒交给 critic”。

注意，$\gamma$ 和 $\lambda$ 都起折扣作用，但含义不同，不应混为一谈：

- $\gamma$ 定义的是**问题**：你有多在乎未来。改变它，就改变了“最优”的含义。
- $\lambda$ 定义的是**估计量**：你有多信任你的 critic。改变它，只会改变你估计同一个目标的方式。

本仓库使用 $\gamma = 0.99$、$\lambda = 0.95$，这是连续控制的标准设置，也是一个不错的默认值。如果你发现自己在调 $\lambda$，真正的问题通常是你的 critic 到底靠不靠谱——先检查 `explained_variance`。

## 4.6 价值目标

critic 需要回归目标。由式 4.6，自然的选择是

$$
\hat G_t = \hat A_t + \hat V(s_t)
\tag{4.8}
$$

这就是 **$\lambda$-回报**。有两点值得注意。

第一，这是一个*自举*目标：它部分由 critic 自己当前的预测构成。让一个网络去拟合由它自身导出的目标，是一个移动目标问题，这也是为什么价值学习会以一种监督回归中不会出现的方式发生振荡。

第二，正是因为使用式 4.8 而不是蒙特卡洛回报 $G_t$，critic 的目标才得以保持低方差。这与前面关于优势的偏差–方差论证是同一回事，只不过应用到了 critic 自己的训练数据上。

代码中：

```python
returns = advantages + self.values
```

只有一行，紧跟在 GAE 循环之后。

## 4.7 再谈截断

式 4.7 在 $d_{t+j} = 1$ 处做掩码。但正如第 1 章所论证的，“回合结束了”和“回报停止了”是两种不同的陈述。

在**终止**处（机器人摔倒了），$V(s_T) = 0$ 是真实成立的，掩码是正确的。

在**截断**处（20 秒时限），回报并没有停止。掩码等于告诉 critic 未来毫无价值，这是错误的，它会让时限附近的每个状态都产生偏差——并且通过贝尔曼递推，还会波及远在时限之前的状态。

正确的处理方式是在边界处把自举值加进奖励里：

$$
\tilde r_t = r_t + \gamma \hat V(s_{t}^{\text{final}}) \quad \text{if truncated at } t
\tag{4.9}
$$

然后照常对优势递推做掩码。在 `ppo_from_scratch/ppo.py` 中，这对应 `truncated_values` 数组以及这一行

```python
rewards += gamma * self.truncated_values
```

Stable-Baselines3 在内部做的是同样的事，依据的是其向量化包装器插入的 `TimeLimit.truncated` info 标志。这一点值得核实：这是原始仓库*没有* bug 的一处，这次检查与那些确实存在的 bug 一起记录在第 14 章中。

## 4.8 优势归一化

还有一个实用步骤，在计算损失之前按小批量（minibatch）进行：

$$
\tilde A_i = \frac{\hat A_i - \operatorname{mean}(\hat A)}{\operatorname{std}(\hat A) + \epsilon}
\tag{4.10}
$$

这一步*并不*来自理论，而且它会引入一点偏差：归一化后的量不再是 $A^\pi$ 的无偏估计。但人们照用不误，而且普遍如此，因为它让有效步长与奖励的尺度无关。

这个性质对本项目尤其重要。第 10 章在调参过程中不断改变奖励权重；如果没有式 4.10，每次改动权重都会重新缩放梯度，进而需要重新调学习率。有了它，两者就解耦了。

减去均值还保证了第 3 章末尾提到的那种失效模式——所有优势同号——不可能在一个小批量内出现。

## 4.9 把它们组合起来

完整的一次 PPO 迭代：

1. 用 $\pi_\theta$ 在 $E$ 个环境中各运行 $N$ 步。存储 $(s_t, a_t, r_t, \log\pi_\theta(a_t|s_t), \hat V(s_t), d_t)$。
2. 为每个存储的步计算 $\delta_t$（式 4.2），并做截断修正（式 4.9）。
3. 反向遍历计算 $\hat A_t$（式 4.6）。
4. 价值目标 $\hat G_t = \hat A_t + \hat V(s_t)$（式 4.8）。
5. 按小批量归一化优势（式 4.10）。
6. 在 PPO 目标上做若干个 epoch 的小批量梯度步——见第 5 章。
7. 丢弃所有数据，回到第 1 步。

第 7 步就是“同策略（on-policy）”的代价：一次昂贵的 rollout 只用于寥寥几次梯度步，然后就被扔掉。第 7 章会讨论这是否值得。

---

**上一章：** [3. 从第一性原理推导策略梯度](03-policy-gradients.md) · **下一章：** [5. 从 TRPO 到 PPO](05-trpo-to-ppo.md)
