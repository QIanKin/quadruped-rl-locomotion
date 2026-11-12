# 6. 逐行解读 PPO

本章从头到尾走读 `ppo_from_scratch/ppo.py`，把每一段代码对应到第 4 章和第 5 章的公式上。阅读时请把这个文件打开放在旁边。

这个文件大约 450 行。其中大约 120 行是算法本身；其余的都是每个真实实现都离不开、却没有任何论文提及的配套机制。这两个数字之间的差距，正是本章的要点。

## 6.1 `RunningMeanStd` —— 观测归一化

```python
class RunningMeanStd:
    def update(self, x): ...
    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)
```

任何一篇 PPO 论文里都没有它。但在实践中它绝对必不可少。

观测中混杂着尺度相差悬殊的各种量：关节角度是 $O(1)$ rad 量级，关节速度可达 20 rad/s，步态时钟则落在 $[-1,1]$ 之间。第一个线性层用同一个权重矩阵作用于所有这些量。如果不做归一化，速度那一块就会主导该层的输出，训练早期便被耗费在学习如何压低它的权重上——网络容量和梯度步数都花在了一个本可以用三行算术就消除的问题上。

更新规则采用 Chan 的并行方差算法，它无需重新遍历数据，就能合并两组（计数、均值、方差）统计量：

$$
\begin{aligned}
\Delta &= \mu_B - \mu_A \\
\mu &= \mu_A + \Delta \frac{n_B}{n_A + n_B} \\
M_2 &= \sigma_A^2 n_A + \sigma_B^2 n_B + \Delta^2 \frac{n_A n_B}{n_A + n_B} \\
\sigma^2 &= M_2 / (n_A + n_B)
\end{aligned}
$$

朴素的替代方案——累加 $\sum x$ 和 $\sum x^2$——在均值相对于方差很大时会出现灾难性的精度损失，而关节角度聚集在 $-1.8$ rad 附近，恰恰就处于这种情形。

有两点操作上的注意事项，本仓库在这两点上都栽过跟头：

- **统计量是模型的一部分。** 一个用归一化观测训练出来的策略，喂给它原始观测就毫无意义。`save()` 有意把它们和权重一起保存。Bug B12（第 14 章）展示的就是两者被分开时会发生什么。
- **评估时冻结。** 对终止观测使用 `_normalize(..., update=False)`，这样评估就不会改变统计量。

## 6.2 `ActorCritic` —— 网络

```python
self.actor  = orthogonal_init(mlp([obs_dim, *hidden, act_dim]))
nn.init.orthogonal_(self.actor[-2].weight, 0.01)
self.critic = orthogonal_init(mlp([obs_dim, *hidden, 1]))
self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
```

**各自独立的主干网络。** 两个网络不共享任何参数。共享在 Atari 相关工作中很常见，因为那里两个头需要相同的视觉特征；但在这里这是个坏主意：critic 必须为一个随机的、会受外力推扰的机器人的回报建模——它需要能编码“我有多大可能摔倒”的特征——而 actor 需要的是一条平滑的控制律。强迫同一个表示同时服务于两者，会让价值损失和策略损失在共享层上互相争夺，而居中调停的只有 $c_v$。

**用 ELU，而不是 ReLU 或 tanh。** ELU 在零点处是平滑的，也没有 ReLU 的死神经元问题。对于一个输出连续关节目标的控制器，平滑性很重要：ReLU 网络的输出是带折点的分段线性函数，这些折点会表现为指令关节轨迹中的微小不连续。这在仿真中基本只是观感问题，到了硬件上则基本不是。

**正交初始化。** 正交矩阵保持输入的范数不变，因此在初始化时，激活值穿过深层堆叠既不会爆炸也不会消失。增益 $\sqrt 2$ 用来补偿经过 ReLU 族非线性后方差预期会减半的效应。

**actor 输出层的增益取 0.01。** 这一条确实很重要，却常被略过。它让初始策略的输出接近零，也就是*名义站立姿态*。再结合 `log_std_init = -1.0`（$\sigma \approx 0.37$），机器人一开始是站着并轻微抖动，而不是把全部十二个关节猛地打到极限。一个在每个回合的前三个控制步就摔倒的智能体，收集不到任何关于行走的信息。

**`log_std` 是一个自由参数**，而不是网络的输出。第 1 章 §1.3 解释了原因：依赖于状态的标准差可能恰恰在困难状态下缩小自身的探索。

### `evaluate`

```python
def evaluate(self, obs, actions):
    dist = self.distribution(obs)
    return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.value(obs)
```

它在**当前**策略下给**已存储**的动作重新打分。这是整个算法的核心：这些动作来自 $\pi_{\theta_\text{old}}$，而式 5.2 需要这些相同动作的 $\log \pi_\theta(a_t \mid s_t)$ 来构造比率。

之所以要 `.sum(-1)`，是因为策略是对角高斯分布：各分量相互独立，所以联合密度是乘积，其对数是求和。忘了这个求和是一个经典 bug——此时比率反映的只是一个关节，而不是整个动作。

## 6.3 `RolloutBuffer` —— 存储与 GAE

形状是 `(n_steps, n_envs, ...)`。每次更新之后，所有数据都会被丢弃；这就是同策略（on-policy）的含义。

### `dones` 约定

```python
self.dones[i] = self._last_done
```

注意，缓冲区把*上一步*的 done 标志和当前观测存放在一起。所以 `dones[t]` 回答的是“`obs[t]` 是不是一个新回合的第一个状态？”，因而 `dones[t+1]` 回答的是“回合是否在第 `t` 步结束？”。这是 CleanRL 的约定，它让 GAE 循环读起来很清爽：

```python
next_non_terminal = 1.0 - self.dones[t + 1]
next_value        = self.values[t + 1]
```

这也是一个很容易写错的差一（off-by-one）问题，而写错的后果是得到一个能训练、能收敛、却微妙地更差的模型——而且任何地方都不会报错。

### `compute_returns`

```python
rewards = self.rewards.copy()
rewards += gamma * self.truncated_values          # Eq 4.9

for t in reversed(range(self.n_steps)):
    delta    = rewards[t] + gamma * next_value * next_non_terminal - self.values[t]   # Eq 4.2
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae              # Eq 4.6
    advantages[t] = last_gae

returns = advantages + self.values                # Eq 4.8
```

六行代码就涵盖了整个第 4 章。有三点值得注意：

1. **倒序计算。** $\hat A_t$ 依赖于 $\hat A_{t+1}$，递推的起点是 rollout 停止处那个状态的价值。
2. **截断自举在循环之前就被折叠进了奖励。** 这是放置它最干净的地方：它让循环本身对这种区别毫无感知，而这正是你想要的，因为这种区别关乎的是*奖励*，而不是递推。
3. **`returns` 是 $\lambda$-回报，而不是蒙特卡洛回报。** 让 critic 拟合它而不是 $G_t$，正是 critic 的目标得以保持低方差的原因。

## 6.4 `collect_rollout`

```python
clipped = np.clip(action_np, -1.0, 1.0)
next_obs, rewards, dones, infos = self.env.step(clipped)
self.buffer.add(self._last_obs, action_np, log_prob.cpu().numpy(), ...)
```

**存储未裁剪的动作，发送裁剪后的动作。** 高斯分布是无界的；动作空间是 $[-1,1]^{12}$。如果你存储的是*裁剪后*的动作，那么更新时 `evaluate()` 计算的是 $\log \pi_\theta(\text{clipped})$，它与存储的 $\log \pi_{\theta_\text{old}}(\text{unclipped})$ 对不上，于是在每一个饱和的维度上，式 5.2 中的比率都会系统性地出错。

由于 `log_std_init = -1.0` 且策略在零附近工作，早期很少出现饱和——这正是这个 bug 难以察觉的原因。它只在策略变得自信之后才会发作，到那时训练会莫名其妙地变得不稳定。

### 截断处理

```python
for i, info in enumerate(infos):
    if dones[i] and info.get("TimeLimit.truncated", False):
        terminal_obs = self._normalize(np.asarray(info["terminal_observation"])[None],
                                       update=False)
        truncated_values[i] = float(self.policy.value(...).item())
```

SB3 的向量化包装器在 `done` 时会自动重置，并把真正的最终观测暂存在 `info["terminal_observation"]` 中。没有它的话，`next_obs` 已经是*重置后*的观测，$V(s_T)$ 就会在错误的状态上求值。使用 `update=False`，是为了不让少量终止观测扭曲滑动统计量。

## 6.5 `update` —— 目标函数

```python
ratio = torch.exp(log_prob - mb["log_probs"])                        # Eq 5.2

adv = mb["advantages"]
adv = (adv - adv.mean()) / (adv.std() + 1e-8)                        # Eq 4.10

surrogate_1  = ratio * adv
surrogate_2  = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
policy_loss  = -torch.min(surrogate_1, surrogate_2).mean()           # Eq 5.6
```

四行代码。这就是 PPO。

那个 `-` 号是因为 PyTorch 做的是最小化，而我们想要最大化 $L^{\text{CLIP}}$。

```python
value_clipped = mb["values"] + torch.clamp(value - mb["values"], -clip_vf, clip_vf)
value_loss = torch.max((value - mb["returns"]) ** 2,
                       (value_clipped - mb["returns"]) ** 2).mean()  # Eq 5.8
```

这里用 `max`，上面用 `min`：对策略，我们取悲观的（更小的）*目标*；对价值，我们取悲观的（更大的）*损失*。两者都是保守的选择；符号之所以相反，是因为一个在被最大化，另一个在被最小化。

```python
loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss   # Eq 5.7
```

其中 `entropy_loss = -entropy.mean()`，所以以正系数加上它，就是在*最大化*熵。

### 梯度裁剪

```python
nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
```

它与 PPO 的裁剪是两回事，承担的也是不同的职责。PPO 的裁剪限制的是*策略分布*能移动多远。而式 5.7 中没有任何东西限制*梯度的大小*——如果某个小批量在做优势标准化时除以了一个极小的标准差，就可能产生巨大的梯度，把整个 rollout 带来的进展一笔勾销。按全局范数裁剪是一份廉价的保险。

### 近似 KL 与提前停止

```python
log_ratio = log_prob - mb["log_probs"]
approx_kl = ((ratio - 1) - log_ratio).mean().item()                   # Eq 5.10
if approx_kl > 1.5 * self.cfg.target_kl:
    early_stopped = True
    break
```

注意这段代码位于 `torch.no_grad()` 之内——它是一个诊断量，而不是目标函数的一部分。

`1.5` 这个倍数是 SB3 的约定，虽然随意但合理：它允许单个噪声较大的小批量通过而不至于放弃整次更新，同时又能拦住真正的漂移。

### 解释方差

```python
out["explained_variance"] = 1 - np.var(y_true - y_pred) / np.var(y_true)
```

$$
\mathrm{EV} = 1 - \frac{\operatorname{Var}[G - \hat V]}{\operatorname{Var}[G]}
\tag{6.1}
$$

1.0 表示 critic 完美地解释了回报。0.0 表示它并不比直接预测均值更好。**负值**表示它比预测均值还差，这在训练早期是一种真实存在且很常见的状态。

这是一次 PPO 训练运行中最具诊断价值的单个数字。第 13 章会解释原因。

## 6.6 `learn` —— 主循环

```python
for update in range(1, n_updates + 1):
    if self.cfg.anneal_lr:
        frac = 1.0 - (update - 1) / n_updates
        for group in self.optimizer.param_groups:
            group["lr"] = frac * self.cfg.learning_rate

    advantages, returns = self.collect_rollout()
    stats = self.update(advantages, returns)
```

学习率线性退火到零。训练后期，优势携带的信号更少（策略已接近局部最优，所以 $A \approx 0$ 再加上噪声），而恒定的学习率会不断把这些噪声注入权重。这个调度并非锦上添花；在这个任务上，它明显减小了训练后期回合回报的抖动。

## 6.7 这里*没有*什么

如实列出生产级实现还会额外加入的东西：

| 特性 | 为何省略 |
|---|---|
| 循环（recurrent）策略 | 观测已经足够马尔可夫；而且会增加大量复杂度 |
| 奖励归一化 | SB3 的 `VecNormalize` 会做这件事；这里的奖励在设计上就已经是 $O(0.5)$ 量级（第 10 章） |
| 多个优化器 | 用一个 Adam 优化全部参数更简单，也管用 |
| 分布式 rollout 工作进程 | `SubprocVecEnv` 借自 SB3；进程管理不是算法本身 |
| 线性以外的学习率调度 | 收益递减 |
| KL 惩罚版 PPO 变体 | 大家用的都是裁剪版 |

## 6.8 验证

真正重要的测试是与 SB3 结果一致：

```bash
python scripts/train.py --n-envs 8 --timesteps 500000 --run-name sb3_ref --seed 0
python ppo_from_scratch/train_scratch.py --n-envs 8 --timesteps 500000 --run-name scratch_ref --seed 0
tensorboard --logdir logs/tensorboard
```

两者的 `rollout/ep_rew_mean` 应当在随机种子带来的噪声范围内彼此吻合。在一次简短的冒烟测试（24k 步，4 个环境）中，两种实现在训练的同一时刻得到的回合回报都在 36–39 之间，这正是预期的吻合程度。

它们不会逐位一致，追求这一点也不值得：SB3 在初始化细节、对最后一个不完整小批量的处理，以及观测归一化的预热上都有所不同。重要的是*曲线*吻合。如果不吻合，差异就出在某个具体的实现细节上，而把它找出来，比重读论文更能让你学懂 PPO。

---

**上一章：** [5. 从 TRPO 到 PPO](05-trpo-to-ppo.md) · **下一章：** [7. 为什么选 PPO 而不是 SAC 或 TD3](07-why-not-sac-td3.md)
