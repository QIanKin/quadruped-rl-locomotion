# 8. Go2、MuJoCo 与 PD 控制

本章讲的是物理系统。它与你使用哪种强化学习算法无关，而一旦把它搞错，算法就无关紧要了——v1 已经证明了这一点。

## 8.1 机器人

Unitree Go2 是一台 15.2 kg 的四足机器人，有十二个主动关节，每条腿三个。这里使用的模型是 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 中的 `unitree_go2/scene.xml`。

模型中的质量分布：

| 部件 | 质量（kg） | ×4 条腿 |
|---|---|---|
| 躯干（`base`） | 6.921 | — |
| 髋 | 0.678 | 2.712 |
| 大腿 | 1.152 | 4.608 |
| 小腿 | 0.241 | 0.965 |
| **总计** | | **15.206** |

将近一半的质量集中在躯干上。这一点很重要：腿相对于身体很轻，所以腿的摆动动力学几乎不会扰动躯干，把身体看作准静态、腿看作快速运动，是一个合理的心智模型。

### 关节及其约定

每条腿有**髋**关节（外展/内收，使腿向侧面转动）、**大腿**关节（髋部屈伸，使腿前后摆动）和**小腿**关节（膝）。

`qpos[7:]` 以及每个动作向量中的关节顺序是 **FL、FR、RL、RR**，每条腿内部为 `(hip, thigh, calf)`。

| 关节 | 范围（rad） | 力矩限幅（N·m） |
|---|---|---|
| 髋（全部四个） | $[-1.047, +1.047]$ | 23.7 |
| 大腿（前腿） | $[-1.571, +3.491]$ | 23.7 |
| 大腿（后腿） | $[-0.524, +4.538]$ | 23.7 |
| 小腿（全部四个） | $[-2.723, -0.838]$ | 45.4 |

有两点值得注意。

**膝关节更强。** 45.4 N·m，其他关节为 23.7，因为膝关节要通过最大的力臂承载机器人的重量。

**小腿的范围不包含零。** $[-2.723, -0.838]$——膝关节*始终*是弯曲的。这台机器人的任何一条腿都不存在伸直（大腿与小腿成一条直线）的构型。记住这一点；它马上就会变得极其重要。

## 8.2 MuJoCo 状态布局

`nq = 19`，`nv = 18`。两者不对称的原因在于四元数：表示姿态需要四个数，但姿态只有三个自由度。

```
qpos[0:3]   base position (world frame), metres
qpos[3:7]   base orientation, unit quaternion, (w, x, y, z)
qpos[7:19]  twelve joint angles, radians

qvel[0:3]   base linear velocity, WORLD frame
qvel[3:6]   base angular velocity, BODY (local) frame
qvel[6:18]  twelve joint velocities
```

`qvel[0:3]` 是世界坐标系下的量。仅这一个事实就构成了 bug B2：v1 把它直接同时送进了奖励和观测，仿佛它就是机器人的前进速度。它不是——它是在固定的世界坐标系下的速度，机器人一旦发生偏航，它就不再对应“前方”了。

转换只需一行：

$$
v_b = R(q)^{\top} v_w
$$

在 `envs/go2_env.py` 中实现为 `quat_rotate_inverse`，利用的恒等式是

$$
q^{*} v q = v - 2w (u \times v) + 2 u \times (u \times v), \qquad u = (q_x, q_y, q_z)
\tag{8.1}
$$

v1 其实有这个函数，实现也正确，却只把它用在了重力上。

## 8.3 `home` 关键帧，以及定义了 v1 的那个 bug

MuJoCo 的 `mj_resetData` 会设置 `qpos = model.qpos0`。对这个模型而言，`qpos0` 就是 MJCF 中各个 body 的位置和关节默认值所隐含的构型——在这里，就是**十二个关节全部精确地处于 0 rad**，躯干位于 $z = 0.445$ m。

模型还定义了一个关键帧：

```xml
<keyframe>
  <key name="home"
       qpos="0 0 0.27  1 0 0 0   0 0.9 -1.8   0 0.9 -1.8   0 0.9 -1.8   0 0.9 -1.8"
       ctrl="0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8"/>
</keyframe>
```

这才是站立姿态：髋 0，大腿 $+0.9$，小腿 $-1.8$，躯干离地 27 cm。

v1 是这样做的：

```python
mujoco.mj_resetData(self.model, self.data)
mujoco.mj_forward(self.model, self.data)
self.default_joint_pos = self.data.qpos[7:].copy()   # <- all zeros
```

然后在每个控制步：

```python
target_pos = self.default_joint_pos + action * self.action_scale   # 0 + a*0.5
```

于是 PD 控制器在每个关节上的目标都是 $0 \pm 0.5$ rad。把它和小腿的合法范围对比一下：

```
reachable calf target:  [-0.5, +0.5]
calf joint limit:       [-2.723, -0.838]
```

**这两个集合没有交集。** 不是“姿态不够理想”，也不是“策略得更努力一些”——对于策略可能输出的每一个动作，指令的膝关节角度都在膝关节的物理行程之外，最少偏出 0.34 rad，最多偏出 1.3。

还有更糟的。`qpos0` 本身的小腿关节就处于 0，这*同样*在 $[-2.723, -0.838]$ 之外。所以 v1 的每一个回合都被重置到一个违反机器人自身关节限位的构型中，MuJoCo 的限位约束立刻开始把关节往回推，而 PD 控制器却往另一个方向拉。每个回合一开始，执行器就在和限位挡块较劲。

检查只需三行，现在它是 `scripts/check_env.py` 中的第一项检查（紧跟在打印模型参数的 “Model” 一节之后）：

```
default_joint_pos : [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]
nominal height    : 0.270 m
matches Go2 home  : YES
```

修复只需一行：

```python
self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
self.default_joint_pos = self.model.key_qpos[self._home_key_id, 7:].copy()
```

这个教训不止适用于本仓库：**加载一个机器人模型时，在做任何其他事情之前，先打印出名义姿态，并对照关节限位检查它。** 这里出错时，什么报错都不会有。仿真照常运行，训练曲线照样上升，论文照样发表。

## 8.4 执行器：力矩，而非位置

Menagerie 中的 Go2 使用 `motor` 执行器：

```xml
<motor class="abduction" name="FL_hip" joint="FL_hip_joint"/>
```

其 `ctrlrange` 以牛·米为单位。所以 `data.ctrl` 是**力矩**，往里写力矩是正确的——这一点值得明说，因为它属于那种看起来像 bug、其实不是的情况。v1 在这一点上做对了。

另一些 MJCF 模型使用 `position` 执行器，此时 `data.ctrl` 是以弧度为单位的目标角度，由 MuJoCo 运行自己内部的伺服控制。往这类执行器里写力矩，会得到一个行为怪异却没有任何错误信息的机器人。一定要检查你的模型用的是哪一种。

## 8.5 PD 控制

策略输出的是**位置目标**，而不是力矩：

$$
q^*_t = q_\text{nom} + \alpha a_t, \qquad \alpha = 0.40\ \text{rad}
$$

再由一个关节空间的 PD 控制律把它们转换为力矩：

$$
\tau = k_p (q^* - q) - k_d \dot q
\tag{8.2}
$$

并按执行器限幅进行裁剪。

### 为什么不让策略直接输出力矩？

有三个理由，按重要性从低到高排列。

**它与硬件相匹配。** 真实 Go2 的底层控制器接收的是带增益的位置目标。力矩策略在部署时得重新推导一遍。

**它是一个条件更好的动作空间。** 位置目标是一个*设定点*；跟踪它所涉及的快速、刚性的动力学由 PD 控制律来处理。策略则在缓慢的、与任务相关的时间尺度上工作。力矩策略还得以 50 Hz 从零开始把刚性的那部分也学会。

**它具有内在的稳定性。** 增益为正的式 8.2 是一个无源的弹簧–阻尼器：即使目标错得离谱，产生的也是有界的、耗散性的力。而错得离谱的力矩会产生什么，就是什么。这让早期训练得以“活下来”。

### 增益选择：靠测量，而不是靠猜

动作为零时，PD 的目标恰好是名义姿态，所以机器人在自身重量下下沉多少，直接衡量了增益是否合理。在这个模型上的测量结果（150 个控制步，然后取平均）：

| $k_p$ | $k_d$ | 稳定高度 | 相对 0.27 m 的下沉量 | 峰值力矩 |
|---|---|---|---|---|
| 20 | 0.5 | 0.196 m | 7.4 cm | 8.8 N·m |
| 25 | 0.6 | 0.206 m | 6.4 cm | 10.4 N·m |
| 40 | 1.0 | 0.242 m | 2.8 cm | 8.0 N·m |
| **55** | **1.4** | **0.254 m** | **1.6 cm** | — |
| 60 | 1.5 | 0.256 m | 1.4 cm | 7.1 N·m |
| 80 | 2.0 | 0.261 m | 0.9 cm | 6.8 N·m |

v2 使用 $k_p = 55$、$k_d = 1.4$。下沉量比乍看之下更重要：它会直接从摆动相可用的抬足高度中扣除，因为当两条腿收起时，另外两条腿要承载整台机器人，下沉得更多（bug B17）。$k_p = 80$ 在纸面上得分更高，但被否决了——刚度那么高时，策略会开始利用任何真实执行器都无法产生的接触冲量。

注意，$k_p = 40$、$k_d = 1.0$ 是 **v1 的取值**，而它们从来都不是问题所在。增益之所以被提高，是出于上面那个单独的、经过单独测量的理由。当一个系统表现不佳时，人们总有把所有东西都改一遍的冲动；而该守住的纪律是，只改证据所指向的部分。

动作尺度通过力矩限幅与增益相互关联：$k_p \times \alpha = 55 \times 0.40 = 22$ N·m，而髋和大腿的限幅是 23.7 N·m，所以在 $|a| = 1$ 时，仅 PD 偏置本身就几乎让执行器饱和了。这就是 $\alpha$ 被限制在 0.40 的原因，进而也限制了步幅和最高速度（bug B20）。

### PD 控制器里*确实*存在的那个 bug

v1 在每个控制步只计算一次力矩，然后保持它不变地推进物理仿真：

```python
torques = self.kp * (target_pos - current_pos) - self.kd * current_vel
self.data.ctrl[:] = np.clip(torques, -ctrl_max, ctrl_max)
for _ in range(self.frame_skip):        # 20 steps = 40 ms
    mujoco.mj_step(self.model, self.data)
```

在 40 毫秒内，$\tau$ 是一个常数。$-k_d \dot q$ 这一项——整个阻尼机制——只被计算一次然后保持不变，这意味着它所依据的速度可能已经过时了长达 40 ms。在足端触地的瞬间，关节速度会在几毫秒内改变符号，此时这个“阻尼”项反而可能在主动*驱动*它本应抑制的振荡。

v2 在循环内部重新计算：

```python
for _ in range(self.decimation):
    torque = kp * (target_pos - self.data.qpos[7:]) - kd * self.data.qvel[6:]
    torque = np.clip(torque, -self.torque_limits, self.torque_limits)
    self.data.ctrl[:] = torque
    mujoco.mj_step(self.model, self.data)
```

这正是真实硬件的做法：底层 PD 回路以 500–1000 Hz 运行，而策略以 50 Hz 运行。用任何其他方式来仿真，仿真的都是一个没有人会部署的控制器。

## 8.6 时序

| | v1 | v2 |
|---|---|---|
| 物理时间步长 | 2 ms | 2 ms |
| decimation（控制降采样） | 20 | 10 |
| 控制频率 | 25 Hz | 50 Hz |
| 每个控制步的 PD 计算次数 | 1 | 10 |
| PD 频率 | 25 Hz | 500 Hz |
| 回合长度 | 1000 步 = 40 s | 1000 步 = 20 s |

50 Hz 是足式机器人强化学习领域的标准。原因在于步态周期：2 Hz 的小跑（trot）在 50 Hz 下每个步态周期有 25 个控制步，而在 25 Hz 下只有 12 个。要安排四只脚的落足，十二个决策点并不算多。

有一个很容易被忽略的后果：**改变控制频率会改变 $\gamma$ 的含义。** 在 $\gamma = 0.99$ 时，有效视界是 $1/(1-\gamma) = 100$ 步——在 25 Hz 下是 4 秒，在 50 Hz 下是 2 秒。控制频率翻倍，规划视界就减半了。两秒对运动控制来说正是合适的数值，所以这是一个令人高兴的后果而不是问题，但它是一个必须加以核实、而不能想当然的后果。

## 8.7 接触

MuJoCo 每一步都会在 `data.contact` 中生成接触。要知道一只脚是否承载了负荷，你必须问两个问题，而 v1 一个都没问：

1. **另一个 geom 是地面吗？** v1 统计了任何涉及足端 geom 的接触，其中包括脚碰小腿、脚碰脚这类自接触。
2. **这个接触承载负荷吗？** v1 用的是接触对是否*存在*。一只轻擦地面的脚和一只承载着机器人四分之一重量的脚，得分完全相同。

v2 两个问题都问了：

```python
if g1 == self.floor_geom_id:   other = g2
elif g2 == self.floor_geom_id: other = g1
else:                          continue
mujoco.mj_contactForce(self.model, self.data, i, force)
if abs(force[0]) < force_threshold:  continue
```

`mj_contactForce` 返回接触坐标系下的一个 6 维向量；`force[0]` 是法向分量。阈值为 1 N，而机器人重 149 N。

v1 还有一个兜底逻辑，用于按名称找不到足端 geom 的情况：

```python
if len(self.foot_geom_ids) != 4:
    self.foot_geom_ids = list(range(self.model.ngeom - 4, self.model.ngeom))
```

“最后四个 geom”可不是足端检测器。如果它真的被触发过，步态奖励就会悄无声息地去奖励任意部件上的接触。v2 改为直接抛出异常：配置错误应该让程序停下来，而不是悄悄改变正在被优化的东西。

地面 geom 的摩擦是 MuJoCo 默认的 `[1.0, 0.005, 0.0001]`，但足端 geom 带有 `priority="1"`，所以足–地接触实际使用的是足端的 `[0.8, 0.02, 0.01]`——分别是滑动、扭转、滚动摩擦。这里最重要的是第一个；训练期间足端和地面的摩擦会乘以一个取自 $[0.4, 1.4]$ 的随机系数（第 12 章）。

## 8.8 完整的控制回路

```
policy(obs)             ->  a in [-1,1]^12          50 Hz
q* = q_nom + 0.40 a                                 50 Hz
    repeat 10x:
        tau = 55 (q* - q) - 1.4 qdot               500 Hz
        clip tau to +/-23.7 (hip, thigh), +/-45.4 (calf)
        mj_step                                     500 Hz
observe, reward, terminate                          50 Hz
```

要记住四个数：50 Hz 的策略，500 Hz 的 PD 和物理仿真，0.40 rad 的动作权限，以及每条腿 $(0, 0.9, -1.8)$ 的名义姿态。接下来三章的所有内容都建立在它们之上。

---

**上一章：** [7. 为什么选 PPO 而不是 SAC 或 TD3](07-why-not-sac-td3.md) · **下一章：** [9. 观测与动作](09-observations-and-actions.md)
