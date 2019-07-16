### 参考
* [David-Silver-Reinforcement-learning](https://github.com/dalmia/David-Silver-Reinforcement-learning): Notes for the Reinforcement Learning course by David Silver along with implementation of various algorithms.
* 强化学习基础 David Silver 笔记
	1. [强化学习概述(Introduction to Reinforcement Learning)](https://zhuanlan.zhihu.com/p/30315707)
	2. [马尔科夫决策过程(MDPs)](https://zhuanlan.zhihu.com/p/30317123)
	3. [动态规划(Planning by Dynamic Programming)](https://zhuanlan.zhihu.com/p/30518290)
	4. [免模型预测(Model-Free Prediction)](https://zhuanlan.zhihu.com/p/30615690)
	5. [免模型决策(Model-Free Control)](https://zhuanlan.zhihu.com/p/31401543)
	6. [值函数近似(Value Function Approximation)](https://zhuanlan.zhihu.com/p/32617897)
	7. [策略梯度(Policy Gradient)](https://zhuanlan.zhihu.com/p/32096947)

### 笔记
#### Lecture 1: Introduction to Reinforcement Learning
> What makes reinforcement learning diﬀerent from other machine learning paradigms?
> * There is no supervisor, only a reward signal
> * Feedback is delayed, not instantaneous
> * Time really matters (sequential, non i.i.d data)
> * Agent’s actions aﬀect the subsequent data it receives

强化学习和机器学习的不同之处：
* 无监督，只有奖赏；
* 反馈延时，非实时；
* 时间很重要（序列化，非独立同分布 i.i.d 数据）；
* Agent 的动作影响接下来接收的序列。

![](https://tuchuang-1252747889.cos.ap-guangzhou.myqcloud.com/2019-06-19-82908E57-8570-494C-9419-307EA417625D-1.jpeg)

$R_t$ 是一个标量的反馈信号，表明在 $t$ 时刻 agent 做的有多好，最大化累积奖赏；

强化学习基于奖赏假设，强化学习的所有的目标可以被描述为最大化期望的累积奖赏。

![](https://tuchuang-1252747889.cos.ap-guangzhou.myqcloud.com/2019-06-19-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-06-19%20%E4%B8%8B%E5%8D%889.24.31.png)

序列化决策的目标是选择做大话总的未来奖赏的动作，动作可能有一个长期的结果，奖赏也可能延迟；有时候可能为了获得更多的长期奖赏而牺牲即时奖赏；

例如金融投资的利润延迟；直升机加油，防止未来的坠毁；禁止反抗运动；

![](https://tuchuang-1252747889.cos.ap-guangzhou.myqcloud.com/2019-06-19-2F243E12-1E5D-4E4D-8043-EFBD7FBBD899.jpeg)

在 $t$ 时刻，Agent 收到来自 Environment 的 $O_t$ 和 $R_t$，执行 $A_t$，Environment 接收 $A_t$ 并反馈给 Agent $R_{t+1}$ 和 $O_{t+1}$，最后进入到下一时刻。

**history** 是所有 observations, actions, rewards 的序列：
$$
H _ { t } = O _ { 1 } , R _ { 1 } , A _ { 1 } , \ldots , A _ { t - 1 } , O _ { t } , R _ { t }
$$

![](https://tuchuang-1252747889.cos.ap-guangzhou.myqcloud.com/2019-06-19-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-06-19%20%E4%B8%8B%E5%8D%889.33.39.png)

一个 Markov state 包含了 history 中所有有用的信息，只需要 $S_t$ 就能作出决策：
$$
\mathbb { P } \left[ S _ { t + 1 } | S _ { t } \right] = \mathbb { P } \left[ S _ { t + 1 } | S _ { 1 } , \ldots , S _ { t } \right]
$$

未来独立于过去，只受当前的影响（一个递进关系，一旦状态已知，history 就能被丢弃）：

$$
H _ { 1 : t } \rightarrow S _ { t } \rightarrow H _ { t + 1 : \infty }
$$

Markov decision process（MDP）是一个 Full observability: agent directly observes environment state，$O _ { t } = S _ { t } ^ { a } = S _ { t } ^ { e }$，即 Agent state 等于 Env state。

而 Partial observability: agent indirectly observes environment，Agent 不直接观测 Env，例如打扑克只能知道公开的牌，被称为 partially observable Markov decision process (POMDP)；这种情况下 Agent 必须构建自己的 state representation $S _ { t } ^ { a }$，可以通过以下方法解决：

![](https://tuchuang-1252747889.cos.ap-guangzhou.myqcloud.com/2019-06-19-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-06-19%20%E4%B8%8B%E5%8D%889.45.06.png)

An RL agent may include one or more of these components:
* Policy: agent’s behaviour function
	* Policy(策略) 是 Agent 的行为，状态到动作的映射(map from state to action)；
	* Deterministic(确定性) policy: 每一种状态有一种确定的决策，$a = \pi ( s )$
	* Stochastic(随机) policy: 每一个状态下该动作(策略)的概率，$\pi ( a | s ) = \mathbb { P } \left[ A _ { t } = a | S _ { t } = s \right]$

* Value function: how good is each state and/or action

	* Value function 用来衡量状态的好坏；预测未来总的奖赏，因此可以选出最好的动作：
$$
v _ { \pi } ( s ) = \mathbb { E } _ { \pi } \left[ R _ { t + 1 } + \gamma R _ { t + 2 } + \gamma ^ { 2 } R _ { t + 3 } + \ldots | S _ { t } = s \right]
$$

* Model: agent’s representation of the environment
	* A model predicts what the environment will do next: model 预测环境接下来会做什么。
	* $\mathcal { P }$ predicts the next state, $\mathcal { P }$ 预测下一个状态。
	* $\mathcal { R }$ predicts the next (immediate) reward, $\mathcal { R }$ 预测下一个即时的奖赏。

$$
\begin{array} { l } { \mathcal { P } _ { s s ^ { \prime } } ^ { a } = \mathbb { P } \left[ S _ { t + 1 } = s ^ { \prime } | S _ { t } = s , A _ { t } = a \right] } \\ { \mathcal { R } _ { s } ^ { a } = \mathbb { E } \left[ R _ { t + 1 } | S _ { t } = s , A _ { t } = a \right] } \end{array}
$$

![](https://tuchuang-1252747889.cos.ap-guangzhou.myqcloud.com/2019-06-19-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-06-19%20%E4%B8%8B%E5%8D%8811.22.07-1.png)

* [DQN从入门到放弃5 深度解读DQN算法](https://zhuanlan.zhihu.com/p/21421729)
* Value Based: [Playing Atari with Deep Reinforcement Learning
](https://arxiv.org/abs/1312.5602)
* [valued based 方法和policy based方法的异同？](https://www.zhihu.com/question/272223357)

强化学习：
* 强化学习是一个 trial-and-error 试错学习；
* Agent 应该从环境的经验之中，找到一个好的策略；
* 不会丢失太多奖赏；

探索和利用：

* Exploration 探索，找到更多的关于环境的信息；
* Exploitation 利用，利用已知信息最大化奖赏；
* 通常探索和利用都很重要；
