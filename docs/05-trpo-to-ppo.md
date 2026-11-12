# 5. 从 TRPO 到 PPO

第 4 章给了我们一个优势估计。第 3 章给了我们一个用到它的梯度。剩下的，才是真正区分能用的强化学习和不能用的强化学习的那个问题：**一次更新应该让策略移动多远？**

## 5.1 为什么步长是头等问题

在监督学习中，步子迈得过大会得到一个糟糕的模型，下一个批次会产生纠正性的梯度，于是你又恢复过来。数据分布是固定的；它不在乎你的模型有多糟。

在强化学习中，是策略*生成数据*。步子迈得太大，就会：

1. 新策略变得更差。
2. 它收集到更差的数据——关于状态空间中更差区域的数据。
3. 这些数据又被用来计算下一次更新。

没有一个固定的分布可以退守。一个退化到足够程度的策略只会收集到机器人在头半秒内就摔倒的轨迹，而这些数据中没有任何梯度能指回行走的方向。这种失败不是收敛缓慢——而是一去不回的崩溃。

因此：要约束更新的大小，而且度量方式要反映*行为*改变了多少，而不是*参数*改变了多少。

这个区别是问题的关键。参数距离毫无意义——在 $\theta$ 中迈出同样的欧氏步长，取决于你所处的位置，既可能让策略几乎不变，也可能把它彻底摧毁。我们关心的是分布 $\pi_{\theta_\text{old}}$ 与 $\pi_\theta$ 之间的距离。

## 5.2 通过重要性采样进行异策略评估

在讲信赖域之前，还需要一个工具。我们想用从 $\pi_{\theta_\text{old}}$ 抽取的样本来估计 $\pi_\theta$ 下的某个量。重要性采样做的正是这件事：

$$
\mathbb{E}_{x \sim q}[f(x)] = \int q(x) f(x) \mathrm{d}x
= \int p(x) \frac{q(x)}{p(x)} f(x) \mathrm{d}x
= \mathbb{E}_{x \sim p}\left[ \frac{q(x)}{p(x)} f(x) \right]
\tag{5.1}
$$

只要在 $q > 0$ 的地方都有 $p > 0$，该式就成立。把它应用到策略梯度上，定义**重要性比率**

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
\tag{5.2}
$$

策略梯度（3.10）就可以用旧数据计算了：

$$
\nabla_\theta J(\theta) \approx \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}}\left[ \sum_t \rho_t(\theta) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat A_t \right]
\tag{5.3}
$$

正是这一点让我们能在每次 rollout 上跑多个 epoch。问题在于：重要性采样只有在 $p$ 和 $q$ 接近时才表现良好。随着 $\pi_\theta$ 偏离 $\pi_{\theta_\text{old}}$，比率会越来越分散，估计量的方差会无界增长——一个 $\rho = 40$ 的样本就能主导整个小批量。

所以我们从两个方向走到了同一个地方。数据崩溃的论证和重要性采样方差的论证说的是同一件事：*让新策略保持在旧策略附近*。

### 关于比率计算的一点说明

永远不要把式 5.2 当作密度之比来计算。对于一个 12 维对角高斯，$\pi(a|s)$ 是十二个密度的乘积，在 float32 下下溢是家常便饭。要在对数空间里计算：

```python
ratio = torch.exp(log_prob - mb["log_probs"])
```

这在数值上是稳定的，因为对数概率的量级是 $O(10)$，而不是 $O(10^{-20})$。

## 5.3 代理目标

定义**保守策略迭代**（conservative policy iteration）目标：

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t\left[ \rho_t(\theta) \hat A_t \right]
\tag{5.4}
$$

注意 $\nabla_\theta L^{\text{CPI}}\big|_{\theta = \theta_\text{old}}$ 等于策略梯度式 3.10，因为 $\rho_t(\theta_\text{old}) = 1$ 且 $\nabla \rho_t = \rho_t \nabla \log \pi_\theta$。所以式 5.4 是 $J$ 的一个*局部*替身：在旧策略处一阶正确，并且可以用旧数据计算。

不加约束地最大化式 5.4 是灾难性的，值得看清楚到底为什么。如果某个样本的 $\hat A_t > 0$，那么随着 $\pi_\theta(a_t \mid s_t)$ 不断增大（对高斯策略而言，这个密度可以无上限地增大），目标会无界增长。优化器会很乐意把策略推向这样一种状态：把全部概率质量都压在这次 rollout 中恰好具有正估计优势的那少数几个动作上——而这些优势估计本身就带噪声，况且原本也只在 $\theta_\text{old}$ 附近才成立。

## 5.4 TRPO：显式的信赖域

信赖域策略优化（Trust Region Policy Optimization，Schulman et al., 2015）用 KL 散度来约束更新：

$$
\begin{aligned}
\max_\theta \quad & \mathbb{E}_t\left[ \rho_t(\theta) \hat A_t \right] \\
\text{s.t.} \quad & \mathbb{E}_t\left[ D_{\mathrm{KL}}\big( \pi_{\theta_\text{old}}(\cdot \mid s_t) \Vert \pi_\theta(\cdot \mid s_t) \big) \right] \le \delta
\end{aligned}
\tag{5.5}
$$

KL 散度是正确的度量，因为它关乎分布，而不是参数。它不随你碰巧采用的策略参数化方式而改变。

TRPO 的理论支撑是实打实的：存在如下形式的单调改进保证

$$
J(\pi_\text{new}) \ge J(\pi_\text{old}) + L^{\text{CPI}}(\pi_\text{new}) - C \cdot \max_s D_{\mathrm{KL}}
$$

因此只要待在信赖域内，就能保证你不会让情况变得更糟。

问题出在这套机制上。求解式 5.5 需要二阶方法：构建 Fisher 信息矩阵（KL 的局部二次近似），借助 Hessian-向量积用共轭梯度算出自然梯度步，再对步长做线搜索以精确满足约束。它能用，但代价高昂，而且繁琐到细小的实现差异都会改变结果。

## 5.5 PPO：同样的思想，靠裁剪实现

近端策略优化（Proximal Policy Optimization，Schulman et al., 2017）提出的问题是：能否只用一阶优化就获得信赖域的效果？

关键洞见在于，我们并不需要*禁止*大步长。我们只需要消除迈大步的*动机*。

$$
\boxed{\quad 
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\Big( \rho_t(\theta) \hat A_t, \quad \quad \operatorname{clip}\big(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon\big) \hat A_t \Big) \right]
\quad }
\tag{5.6}
$$

通常取 $\varepsilon = 0.2$。逐个情形地读它——这是值得能凭记忆重新推出来的部分。

**情形 $\hat A_t > 0$**（该动作优于平均水平；我们希望它更可能被选中，即希望 $\rho$ 增大）：

- $\rho < 1 + \varepsilon$：min 选中未裁剪的分支 $\rho \hat A$（在 $[1-\varepsilon, 1+\varepsilon]$ 内两个分支相等；$\rho < 1-\varepsilon$ 时被裁剪的分支更大），梯度正常流动。
- $\rho > 1 + \varepsilon$：被裁剪的分支是 $(1+\varepsilon)\hat A$，一个常数。min 会选中它（它更小）。关于 $\theta$ 的梯度为**零**。

所以一旦该动作的概率已经是原来的 $1+\varepsilon$ 倍，再让它变得更可能也不会得到任何额外奖励。不是惩罚——而是不再有激励。

**情形 $\hat A_t < 0$**（我们希望 $\rho$ 减小）：

- $\rho > 1 - \varepsilon$：min 选中未裁剪的分支 $\rho \hat A$（$\rho > 1+\varepsilon$ 时，被裁剪的分支 $(1+\varepsilon)\hat A$ 比 $\rho\hat A$ 更大），梯度流动。
- $\rho < 1 - \varepsilon$：被裁剪的分支是 $(1-\varepsilon)\hat A$，由于 $\hat A < 0$，它比 $\rho\hat A$ *更小*。min 选中它，关于 $\theta$ 的梯度为**零**。

所以一旦该动作的概率已经降到原来的 $1-\varepsilon$ 倍，再继续压低它也不会得到任何额外奖励。

这种不对称是有意为之的，也正是使用 `min` 而不是单纯裁剪的意义所在：裁剪只在比率朝着*让目标改善*的方向越界时才生效。如果比率朝着让目标*变差*的方向移动（$\hat A > 0$ 而 $\rho < 1-\varepsilon$，或 $\hat A < 0$ 而 $\rho > 1+\varepsilon$），min 选中的是未裁剪的分支，梯度照常流动，策略总能把错误的移动纠正回来。悲观总是安全的；需要约束的是乐观。

### 悲观视角

看待式 5.6 的另一种方式：$L^{\text{CLIP}}$ 是 $L^{\text{CPI}}$ 的一个**下界**。`min` 总是从两个估计中选出更保守的那个。PPO 最大化的是改进量的一个悲观界，这就是为什么即使优势估计很差，它也往往很稳定。

## 5.6 完整目标

实践中，有三项被联合优化：

$$
L(\theta, \psi) = \underbrace{-L^{\text{CLIP}}(\theta)}_{\text{policy}}
+ c_v \underbrace{L^{V}(\psi)}_{\text{value}}
- c_e \underbrace{\mathbb{E}_t\big[ \mathcal{H}[\pi_\theta(\cdot \mid s_t)] \big]}_{\text{entropy}}
\tag{5.7}
$$

（取了负号，因为框架做的是最小化。）

### 价值损失，以及对它也做裁剪

$$
L^{V}(\psi) = \mathbb{E}_t \Big[ \max\Big( (\hat V_\psi(s_t) - \hat G_t)^2, \quad (\hat V^{\text{clip}}_\psi(s_t) - \hat G_t)^2 \Big) \Big]
\tag{5.8}
$$

$$
\hat V^{\text{clip}}_\psi(s_t) = \hat V_{\psi_\text{old}}(s_t) + \operatorname{clip}\big( \hat V_\psi(s_t) - \hat V_{\psi_\text{old}}(s_t), -\varepsilon_v, +\varepsilon_v \big)
$$

动机与策略裁剪相同。critic 拟合的目标是由它自己的旧预测构建的（式 4.8），所以一次大幅度的 critic 更新会让它所拟合的那些目标本身变得过时，而本次更新剩下的 epoch 就是在针对一个已经不存在的价值函数做优化。`max` 同样选中悲观的那个分支。

这里 $c_v = 0.5$。这个系数在 actor 学习与 critic 学习之间做权衡；太大，共享的优化器预算就会流向 critic；太小，critic 会跟不上，优势就成了噪声。

### 熵奖励项

$$
\mathcal{H}[\pi] = -\int \pi(a \mid s) \log \pi(a \mid s) \mathrm{d}a
$$

对于对角高斯，它有闭式解：

$$
\mathcal{H} = \sum_{i=1}^{n} \left( \log \sigma_i + \tfrac{1}{2}\log(2\pi e) \right)
\tag{5.9}
$$

所以最大化熵就是单纯地把 $\log\sigma$ 往上推。它对抗的是过早坍缩：没有它，策略的方差会单调收缩（方差越低意味着坏动作越少，意味着即时回报越高），探索在找到好步态之前就消亡了。

这个系数比看上去更重要。v1 用的是 $c_e = 0.01$；v2 用的是 0.002。在 $c_e = 0.01$ 时，由于 v2 每步奖励的量级只有约 $0.05$（按 $\Delta t$ 缩放后），熵奖励项在目标中占了很大比例，策略等于是被花钱雇来保持噪声——对足式机器人来说，这意味着它永远无法稳定进入小跑所需的那种干净、低方差的极限环。症状是策略一直在走，却始终显得抽搐，日志里的 `std` 迟迟降不下来。

## 5.7 近似 KL 与提前停止

即便有了裁剪，*度量*一下策略移动了多远仍然是值得的。朴素估计量

$$
\widehat{\mathrm{KL}} = \mathbb{E}_t[-\log \rho_t]
$$

是无偏的，但在有限样本上可能为负，这对一个散度来说有点尴尬。Schulman 提出的低方差替代方案是：

$$
\widehat{\mathrm{KL}} = \mathbb{E}_t\left[ (\rho_t - 1) - \log \rho_t \right]
\tag{5.10}
$$

它对每个样本都非负，因为对所有 $x>0$ 都有 $x - 1 - \log x \ge 0$，且等号在 $x=1$ 时成立。它的方差也更低。SB3 的日志和我们的日志中出现的 `approx_kl` 就是它。

PPO 的实现通常用它来**提前停止**一次更新：如果测得的 KL 超过目标值的某个倍数（这里是 $1.5 \times 0.02$），就放弃剩余的 epoch。理由就是 5.2 节中的重要性采样论证——过了那个点，收集到的数据已经太过异策略，无论裁剪与否都无法再从中学习。

```python
if approx_kl > 1.5 * self.cfg.target_kl:
    early_stopped = True
    break
```

## 5.8 为什么多个 epoch 竟然行得通

PPO 在一次 rollout 上做 $K$ 个 epoch 的小批量 SGD。第 3 章说过，策略梯度只对收集数据的那个策略有效。那为什么这样做不是彻头彻尾的错误？

因为式 5.6 并不是策略梯度——它是一个代理目标：借助重要性比率，它对 $\theta_\text{old}$ 附近的策略*依然有意义*；而一旦策略漂移超出 $1 \pm \varepsilon$，它就*不再奖励移动*。第一个 epoch 几乎是同策略的；到第五个 epoch 时，许多样本已被裁剪，完全不贡献梯度，而这恰恰是预期中的行为。这个算法会优雅地退化为什么都不做，而不是去做错事。

这就是为什么 `clip_fraction` 是一个有用的诊断指标，也是为什么它应当非零。见第 13 章。

## 5.9 超参数，以及本仓库的取值

| 符号 | 名称 | v2 取值 | v1 取值 | 说明 |
|---|---|---|---|---|
| $\varepsilon$ | `clip_range` | 0.2 | 0.2 | 标准取值；很少值得改动 |
| $\varepsilon_v$ | `clip_range_vf` | 0.2（仅 `ppo_from_scratch`） | 无 | 价值裁剪，v2 中新增；`scripts/train.py` 未传入该参数，SB3 默认 `None`，即不裁剪 |
| $c_v$ | `vf_coef` | 0.5 | 0.5 | |
| $c_e$ | `ent_coef` | 0.002 | 0.01 | v1 的取值高到足以让策略无法稳定下来 |
| $K$ | `n_epochs` | 5 | 10 | |
| $N$ | `n_steps` | 512 | 2048 | 每个环境 |
| $E$ | `n_envs` | 16 | 8 | |
| | rollout | 8192 | 16384 | $N \times E$ |
| $B$ | `batch_size` | 2048 | 64 | |
| | 每次 rollout 的梯度步数 | 20 | **2560** | $K \cdot N E / B$ |
| | `target_kl` | 0.02 | 无 | |
| | `max_grad_norm` | 1.0 | 0.5 | |

最值得盯着看的一行是**每次 rollout 的梯度步数**。v1 在单次 rollout 的数据上做了 2560 次梯度步。无论怎样裁剪，都不可能让策略在 2560 次更新中始终保持在 $\theta_\text{old}$ 附近；裁剪比例会饱和，大多数样本不贡献任何东西，算力白白浪费。它既是稳定性隐患，又是巨大的时间浪费。见第 14 章，B14。

## 5.10 完整算法

```
initialise theta, psi
for iteration = 1, 2, ...:
    # --- collect ---
    for t = 1..N in each of E environments:
        a_t ~ pi_theta(. | s_t)
        step the environment, store (s, a, r, logp, V, done)

    # --- estimate (chapter 4) ---
    delta_t  = r_t + gamma V(s_{t+1})(1 - d_{t+1}) - V(s_t)
    A_t      = delta_t + gamma lambda (1 - d_{t+1}) A_{t+1}      backwards
    G_t      = A_t + V(s_t)

    # --- optimise (this chapter) ---
    for epoch = 1..K:
        for each minibatch:
            rho    = exp(logpi_theta(a|s) - logp_old)
            A      = (A - mean A) / std A
            L_clip = -min(rho A, clip(rho, 1-eps, 1+eps) A)
            L_v    = max((V - G)^2, (V_clip - G)^2)
            L      = L_clip + c_v L_v - c_e H
            step Adam on L, with gradient norm clipped
            if approx_kl > 1.5 * target: stop
```

这就是整个算法。`ppo_from_scratch/ppo.py` 就是这些，再加上观测归一化和一些簿记工作。第 6 章会逐行讲解它。

---

**上一章：** [4. Actor-Critic 与 GAE](04-actor-critic-and-gae.md) · **下一章：** [6. 逐行解读 PPO](06-ppo-line-by-line.md)
