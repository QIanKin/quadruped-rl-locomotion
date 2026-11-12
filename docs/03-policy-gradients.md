# 3. 从第一性原理推导策略梯度

本章完整地推导 $\nabla_\theta J(\theta)$。这是全书的数学核心：PPO 是对这里推导出的结果的一种修改，只要你跟上了本章，剩下的就都是细节。

## 3.1 障碍

我们想要最大化

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[ R(\tau) ],
\qquad R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t
$$

而要做梯度上升，我们需要 $\nabla_\theta J$。

困难立刻就出现了。把期望写成积分：

$$
J(\theta) = \int p_\theta(\tau) R(\tau) \mathrm{d}\tau
$$

于是

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) R(\tau) \mathrm{d}\tau
\tag{3.1}
$$

$\theta$ 出现在**我们采样所用的分布**里，而不是出现在被平均的量里。式 3.1 不再是关于 $p_\theta$ 的期望——它是对 $\nabla_\theta p_\theta$ 的积分，而后者不是概率分布，我们也无法从中采样。我们没法靠运行策略来估计它。

## 3.2 对数导数技巧

出路只需一行微积分。对任意正的 $p$，

$$
\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
\quad\Longrightarrow\quad
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
\tag{3.2}
$$

代入 3.1：

$$
\nabla_\theta J(\theta)
= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \mathrm{d}\tau
= \mathbb{E}_{\tau \sim p_\theta}\left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]
\tag{3.3}
$$

现在它是一个关于我们确实能生成的轨迹的期望了。运行策略，收集轨迹，对 $\nabla_\theta \log p_\theta(\tau) R(\tau)$ 求平均。

这个技巧有个名字——**得分函数估计量**（score function estimator），或称 REINFORCE——值得体会一下它以多么小的代价换来了多么大的收益。它把“期望的梯度”变成了“梯度的期望”，而这正是“不可能”与“一个 for 循环”之间的差别。

## 3.3 动力学被消掉了

式 3.3 看起来仍然没法用：$\log p_\theta(\tau)$ 里包含转移核 $P$，而我们并不知道它。利用式 1.3 把它展开：

$$
\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} \Big[ \log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t) \Big]
$$

现在对 $\theta$ 求导。初始状态项 $\log p(s_0)$ 不依赖于 $\theta$。任何 $\log P$ 项也不依赖于它——物理规律可不在乎我们的网络权重是多少。两者都消失了：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\tag{3.4}
$$

**这就是无模型强化学习之所以可行的原因。** 我们从来不需要 $P$。梯度对环境的依赖，仅仅体现在采样到的轨迹碰巧获得的奖励上。把 3.4 代入 3.3：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau}\left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) R(\tau) \right]
\tag{3.5}
$$

这就是 **REINFORCE**（Williams，1992）。它是策略梯度的一个正确的、无偏的估计量，但在实践中几乎毫无用处。接下来两节解释原因，并修复它。

## 3.4 问题一：未来无法改变过去

在 3.5 中，轨迹里的每一个动作都由**总**回报 $R(\tau)$ 加权，其中包括在该动作被采取*之前*就已获得的奖励。$t = 300$ 时的动作，被记上了 $t = 5$ 时挣到的奖励。

这除了噪声什么也贡献不了，而且我们可以证明这一点。考虑某个较早的奖励 $r_{t'}$ 对一个较晚动作 $a_t$（$t > t'$）的梯度项的贡献：

$$
\mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot r_{t'} \right]
$$

以直到 $s_t$（含）为止的一切为条件。那么 $r_{t'}$ 就是一个固定的数，并且

$$
\mathbb{E}_{a_t \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
= \int \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} \mathrm{d}a
= \nabla_\theta \int \pi_\theta(a \mid s_t) \mathrm{d}a
= \nabla_\theta 1 = 0
\tag{3.6}
$$

得分的期望为零。所以这个交叉项的期望为零：它是无偏的噪声，纯粹的方差，没有任何信号。去掉它，就得到了**未来回报**（reward-to-go）形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right],
\qquad G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
\tag{3.7}
$$

式 3.6 本身就值得记住。它马上就会再次用到，在第 5 章还会再用一次。

## 3.5 问题二：基线

即便用了未来回报，方差依然残酷。对这台机器人来说，$G_t$ 是一百多个带噪项之和；*同一个*策略从*同一个*状态出发跑两次，由于接触的混沌性，结果可能相差很大。

修复办法再次用到 3.6。减去任意一个不依赖于 $a_t$ 的函数 $b(s_t)$：

$$
\mathbb{E}_{a_t}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right]
= b(s_t) \mathbb{E}_{a_t}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = 0
$$

因此对于**任意**依赖于状态的基线 $b$：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \big( G_t - b(s_t) \big) \right]
\tag{3.8}
$$

无论 $b$ 怎么选，这个估计量都保持无偏，但它的**方差会变**。选好 $b$，就是白捡的方差缩减。

### 最优基线

关于 $b$ 最小化估计量的方差。记 $g = \nabla_\theta \log \pi_\theta(a \mid s)$，并且一次只处理一个分量，

$$
\operatorname{Var}\left[ g (G - b) \right] = \mathbb{E}\left[ g^2 (G-b)^2 \right] - \left( \mathbb{E}[g(G-b)] \right)^2
$$

第二项不依赖于 $b$（它就是那个无偏梯度）。令第一项的 $\partial / \partial b$ 为零：

$$
-2\mathbb{E}\left[ g^2 (G - b) \right] = 0
\quad\Longrightarrow\quad
b^\star = \frac{\mathbb{E}\left[ g^2 G \right]}{\mathbb{E}\left[ g^2 \right]}
\tag{3.9}
$$

——一个以得分加权的平均回报。实践中没人用 3.9：它需要逐参数的基线，还需要得分的二阶矩。接近最优的实用选择，是从该状态出发的不加权平均回报，而这恰好就是

$$
b(s) = V^\pi(s)
$$

而选定它之后，$G_t - V^\pi(s_t)$ 就是 $Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$ 的一个无偏样本。第 2 章的优势函数出现了——它不是作为一种启发式，而是作为最优方差缩减的自然结果出现的：

$$
\boxed{\quad 
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) A^\pi(s_t, a_t) \right]
\quad }
\tag{3.10}
$$

这就是每个现代算法所使用的那种形式的**策略梯度定理**。

## 3.6 解读这个梯度

式 3.10 有一个值得内化的解读，因为它能解释你将遇到的每一种训练病症。

$\nabla_\theta \log \pi_\theta(a \mid s)$ 指向参数空间中让 $a$ 在 $s$ 中**更可能**被选中的方向。乘以 $A$ 之后：

- $A > 0$（比策略的平均水平好）：沿该方向迈一步。让这个动作更可能发生。
- $A < 0$：逆着它迈一步。让这个动作更不可能发生。
- $A \approx 0$：不更新。

所以策略梯度就是：*多做有效的事，少做无效的事，而“有效”的意思是“胜过了我自己的平均水平”。*

两个推论：

**如果 critic 有偏差，策略就会学到这个偏差。** $A$ 被估计为 $G_t - \hat V(s_t)$。如果 $\hat V$ 在某个区域系统性地低估价值，那么该区域中的每个动作都会得到正的优势，不论好坏都会被强化。这就是为什么 `explained_variance` 是要盯住的那个数。

**如果所有优势的符号都相同，就不会发生任何有用的事。** 策略会一律朝它采取过的每个动作推进，这基本上只会让熵膨胀或收缩。实践中，优势会按小批量（minibatch）进行标准化（见 `ppo_from_scratch/ppo.py` 中的 `normalize_advantage`），部分原因就是为了保证这种情况不会发生。

## 3.7 你在 PyTorch 里实际写的损失

自动微分框架做的是最小化，而且对一个标量求导。你不会直接把式 3.10 写成代码——你写的是一个代理标量，它的梯度等于式 3.10：

$$
L(\theta) = - \frac{1}{N} \sum_{i} \log \pi_\theta(a_i \mid s_i) \hat A_i
\tag{3.11}
$$

其中 $\hat A_i$ 被当作**常数**（从计算图中分离出来）。于是 $\nabla_\theta L$ 恰好是 3.10 的相反数，对 $L$ 做梯度下降就是对 $J$ 做梯度上升。

$L$ 在任何有意义的层面上都不是一个损失。它的值不衡量性能；它可正可负，也不会随着训练变好而下降。看着 `policy_loss` 往下走，什么也说明不了。这是对强化学习训练日志最常见的一种误读。

## 3.8 为什么光有 REINFORCE 还不够

到目前为止的一切都是正确的，但仍然不够。还剩三个问题，它们决定了接下来两章的内容。

**1. $A^\pi$ 是未知的。** 式 3.10 需要真实的优势。使用蒙特卡洛估计 $G_t - \hat V(s_t)$ 是无偏的，但方差很高；使用单步 TD 估计方差很低，但会因 critic 的误差而有偏。第 4 章（GAE）给出了有原则的折中插值。

**2. 走一步之后数据就过时了。** 式 3.10 是在 $\pi_\theta$——*当前*策略——下的期望。只要你走了一步梯度，你收集到的轨迹就变成由另一个策略生成的了，估计量也就不再有效。因此，严格的同策略（on-policy）学习每次 rollout 只允许一次更新，这是对昂贵仿真的灾难性浪费。第 5 章（重要性采样与 PPO）讲的就是如何从一次 rollout 中得到多次更新，而不让估计量崩溃。

**3. 步长没有上限，后果也不对称。** 在监督学习中，步子太大只会给出一个糟糕的预测，下一个批次就会把它纠正回来。在强化学习中，步子太大会改变策略，而策略又会改变你接下来收集的*数据分布*。一个变得足够差的策略只会收集到关于摔倒的数据，而从那里不存在通往胜任的梯度。这就是为什么强化学习需要信赖域而监督学习不需要——同样见第 5 章。

---

**上一章：** [2. 价值函数与策略](02-value-functions-and-policies.md) · **下一章：** [4. Actor-Critic 与 GAE](04-actor-critic-and-gae.md)
