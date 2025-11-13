# 训练实现说明：

### 任务说明

我们需要训练一个diffusion language model，使其获得按照逻辑顺序进行解码的能力。数独作为逻辑性非常强的任务，能够很好的确定解码逻辑顺序，因此本项目主要利用数独任务来提升dllm（以llada为例）的解码顺序逻辑性。

首先从一个最开始的训练目标出发：我们希望模型针对一个问题输出的结果中，大部分情况都能够按照逻辑顺序进行解码。而当模型填入一个错误的数之后，后面的状态都不再有意义：因此出于训练效率的考虑，我们需要将模型从输出错误的状态处阶段。

由于训练目标是一个离散的值，因此我们能够写出一个强化学习训练目标。通过推导，我们分离出了两块训练：一块类似SFT，一块类似RL。希望最后的训练过程能够将这两部分同时在线训练（学习率按理说可能相同比较好，但也可以采用不同学习率的方式）。

### 最后训练目标

$$
\begin{align*}
\mathcal J(\theta)=&\mathbb E_{
q\sim \mathcal D,\ 
\{\tau_{j,1:\big(T-h(q)\big)}^{1:L}\}_{j=1}^G
\sim
\pi_{\theta_{\text{old}}}(\cdot|q)
}
\frac{1}G\sum_{j=1}^G
\sum_{t=1}^{T-h(q)-1}\min \big\{R_\theta(t,\tau_j, q) \mathcal A(\tau_j, t,q),\text{clip}(R_\theta(t,\tau_j,q),1-\epsilon,1+\epsilon)\mathcal A(\tau_j, t,q) \big\} \\

&+

\mathbb E_{
q\sim \mathcal D,\ 
\tau_{1:\big(T-h(q)\big)}^{1:L}
\sim
\pi_{\theta_{\text{old}}}(\cdot|q)
}\sum_{t=1}^{T-h(q)}\text{sg}[R_\theta(t, \tau, q)]\cdot f_\theta(\tau_{t}^{1:L},q) 
\end{align*}
$$

h(q)表示预填的token数量，也即我需要模型解决一个数独问题，那么我会将后续的mask序列中能够确定的位置填上具体的token（比如我会将问题中明确给出的token填入mask序列中，以免模型中这些位置上输出其他token导致错误）。此外：
$$
R_\theta(i,\tau,q)

=\frac{\pi_\theta(\tau_{i}^{1:L}|\tau_{i+1}^{1:L},q)}{\pi_{\theta_\text{old}}(\tau_{i}^{1:L}|\tau_{i+1}^{1:L},q)}

=
\frac{\pi_\theta(\ell(\tau,i)|\tau_{i+1}^{1:L},q)}{\pi_{\theta_\text{old}}(\ell(\tau,i)|\tau_{i+1}^{1:L},q)}

\cdot

\frac{\pi_\theta(\tau_i^{\ell(\tau,i)}|\ell(\tau,i),\tau_{i+1}^{1:L},q)}{\pi_{\theta_\text{old}}(\tau_i^{\ell(\tau,i)}|\ell(\tau,i),\tau_{i+1}^{1:L},q)}
$$

$\pi_\theta(\ell(\tau,i)|\tau_{i+1}^{1:L},q)$这种就是选择解码位置，后面那项$\pi_\theta(\tau_i^{\ell(\tau,i)}|\ell(\tau,i),\tau_{i+1}^{1:L},q)$是选择解码成哪个token。此外给式中的$\ell$做一个定义：$\ell(\tau,t)\text{ s.t. }\tau_t^{(\ell)}\neq\mathbf m,\ \tau_{t+1}^{(\ell)}=\mathbf m$，表示$\tau$这个轨迹从第t+1时间步到第t时间步解码的token的位置。
$$
r(\tau, t,q)=\sum_{i=1}^t \gamma^{t-i}\text{sg}[f_\theta(\tau_i^{1:L},q)] \\
\mathcal A(\tau, t, q)=\frac{r(\tau,t,q)-\text{mean}\big(\{r(\tau_j,t,q)\}_{j=1}^G\big)}{\text{std}\big(\{r(\tau_j,t,q)\}_{j=1}^G\big)}
$$

$$
f_\theta(\tau_{t+1}^{1:L}, q)=\lambda\sum_{\ell(\tau,t)\in S_2}\pi_\theta(\ell(\tau,t)|\tau_{t+1}^{1:L},q)+\sum_{\ell(\tau,t)\in S_1}\pi_\theta(\ell(\tau,t)|\tau_{t+1}^{1:L},q)\cdot\pi_\theta(a^*|\ell(\tau,t),\tau_{t+1}^{1:L},q)
$$

$\ell(\tau,t)$可以分成两类：第一类$S_1$，表示位置在sudoku map内部且能100%确定token（数独任务保证中每一个状态都有至少一个位置能够根据已有信息100%确定）；第二类$S_2$表示在map的外部。$a^*$表示根据问题$q$解出来在位置$\ell(\tau,t)$​应该填入的token。sg就是detach操作。

上式中优势$\mathcal A$的含义就是：对于同一个问题采样得到的不同trajectory，对于一个确定的时间步，将不同轨迹中这个时间步上得到的reward——也即$r(\tau, t, q)$——进行归一化。

此外，注意在计算$f_\theta$的时候，使用logsumexp操作。具体而言，是对于求和项进行拆解，对每一项先log，然后依次执行exp，sum，log操作。最后得到计算结果再exp即可得到原本的sum操作。这样做的好处是：能够对各个数在log域计算，然后最后恢复原数。logsumexp有一些细节优化，已经被torch打包成具体函数。

选择token位置的建模方法：
$$
\pi_\theta(\ell(\tau,t)|\tau_{t+1}^{1:L},q)=\frac{
\exp\big(\log\frac{\text{top}_1 \pi_\theta(\cdot|\ell(\tau,t),\tau_{t+1}^{1:L},q)+\epsilon}{\text{top}_2 \pi_\theta(\cdot|\ell(\tau,t),\tau_{t+1}^{1:L},q)+\epsilon}/\Tau\big)
}{
\sum_l\exp\big(\log\frac{\text{top}_1 \pi_\theta(\cdot|l,\tau_{t+1}^{1:L},q)+\epsilon}{\text{top}_2 \pi_\theta(\cdot|l,\tau_{t+1}^{1:L},q)+\epsilon}/\Tau\big)
}
$$
这个$\epsilon$是一个很小的数比如1e-9，不是clip参数。

确定位置后，选择token的建模方法就不用说了：就是传统的transformer前向传播，得到每一个位置的logits，然后根据确定位置的词表形状的logits来计算分布然后采样即可。

### 代码编写说明

训练超参数：

平衡不同位置reward权重的系数$\lambda$（可初始化0.5），折扣因子$\gamma$（可初始化0.95），两个clip参数$\epsilon$（可初始化1e-9），两个温度（一个选取位置，一个选择解码成的token），分别对应前面的$\pi_\theta(\ell(\tau,t)|\tau_{t+1}^{1:L},q)$和$\pi_\theta(\tau_i^{\ell(\tau,i)}|\ell(\tau,i),\tau_{i+1}^{1:L},q)$的采样系数分别可初始化为0.05，0.6。

训练过程中需要记录用于调试的指标（记录他们随着训练step数量的变化）：

1. clip触发的比例
2. 采样位置分布的熵（把整个采样序列的每一个step的熵计算平均值）
3. 贪心采样位置（margin最大的位置）位于sudoku map之内的情况中，该贪心位置是100%能够确定的位置的占比（也即在S1内）
4. 采样位置分布的最大概率值（用于调整softmax的温度）

注意事项：

1. 可以给RL损失项与SFT损失项分别设定不同的学习率
2. 训练数据格式：只有一个集合，放了很多的prompt，描述了数独问题
3. 提供两个函数：
   1. 预填函数 pre_fill：接收问题的prompt, 模型tokenizer，最大生成长度。返回结果：一个实现预填到最大生成长度的token id序列，被填入的token的index（从prompt后第一个token编号为0开始），prompt的长度（也即prompt后第一个token的绝对位置编号），sudoku map的范围（一个tuple，比如(a,b), 说明sudoku的范围是[a, b)。也是从prompt后第一个token开始编号）
      该函数会将system prompt，response开头的说明文字包含进来。并且response的长度等于最大生成长度。
   2. 100%确定位置计算 detect_definite：接收一个解码的中间状态（token id序列，可能需要重新decode成自然语言）。返回结果：能够100%确定的index，规则和1相同
   3. 判断当前状态是否错误的函数 judge_error：接收一个解码的中间状态，然后返回一个bool值，表示当前状态是否已经出现了填错数字的状态。
4. 如果采样过程中发现模型已经填入了一个错误的token，那就终止采样过程。对于被终止的序列，后续的step由于需要计算advantage，还需要给reward（也即f_theta）。对于这种情况，直接给0。然后在SFT损失项中，这些被赋值为0的部分将不会产生梯度。
5. 注意：公式中的降噪过程是从大t到小t，而代码实现中的采样轨迹是从大的噪声到小的噪声。也即公式中从小t到大t代表代码中的反向轨迹。