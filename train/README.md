现有在扩散大语言模型上的主流强化学习方法主要采用如下优化目标（以TraceRL为代表）：
$$
\mathcal J(\theta)=\mathbb E_{
q\sim \mathcal D,\ 
\{\tau_{j}^{1:L}\}_{j=1}^G
\sim
\pi_{\theta_{\text{old}}}(\cdot|q)
}
\frac{1}G\sum_{j=1}^G
\sum_{t=1}^{L}\min \big\{R_\theta(t,\tau_j, q) \mathcal A(\tau_j, t,q),\text{clip}(R_\theta(t,\tau_j,q),1-\epsilon,1+\epsilon)\mathcal A(\tau_j, t,q) \big\} \\
R_\theta(i,\tau,q)

=\frac{\pi_\theta(\tau_{i}^{1:L}|\tau_{i+1}^{1:L},q)}{\pi_{\theta_\text{old}}(\tau_{i}^{1:L}|\tau_{i+1}^{1:L},q)}
$$

由于扩散大语言模型特殊的解码机制，使得每一步解码（从$\tau_{i+1}^{1:L}$变为$\tau_i^{1:L}$）并不像ar模型直接在原本生成的序列后解码next token，而是会在后续的完整解码范围内选择一个未被解码的[mask] token进行解码。现有方法将每一步解码的概率建模如下：
$$
\pi_\theta(\tau_{i}^{1:L}|\tau_{i+1}^{1:L},q)=\pi_\theta(\tau_t^{\ell(\tau,t)}|\ell(\tau,t),\tau_{t+1}^{1:L},q)
$$
其中，$\ell(\tau,t)\text{ s.t. }\tau_t^{(\ell)}\neq\mathbf m,\ \tau_{t+1}^{(\ell)}=\mathbf m$。但这种建模过程仅仅考虑了选定了某个特定位置后根据该位置的logits解码出特定token的过程，忽略了选定解码位置这一动作的概率，因此当前方法对于扩散语言模型的解码动作序列建模是不完整的。

现有用于选定解码位置的方法主要是基于贪心而非随机，主流的贪心策略例如：
$$
\ell(\tau,t)=\text{argmax}_l\big(\max_a \pi_\theta(a|l,\tau_{t+1}^{1:L},q)\big)
$$
因此为了优化该贪心过程，需要将其近似为可微的参数化随机采样。因此我们提出如下解码的概率建模方式：
$$
\pi_\theta(\tau_{i}^{1:L}|\tau_{i+1}^{1:L},q)=\pi_\theta(\ell(\tau,t)|\tau_{t+1}^{1:L},q)\cdot\pi_\theta(\tau_t^{\ell(\tau,t)}|\ell(\tau,t),\tau_{t+1}^{1:L},q)
$$
通过将解码过程显示拆分成两个部分，能够同时优化选位置与选token两种行为，扩大优化范围。

## 新增训练选项

- **KL散度惩罚（`training.kl_coefficient`）**：基于新旧策略对解码token概率的比值加入$x-\log x-1$形式的惩罚，默认系数为0.04，可通过配置调整。
- **单样本多轮训练（`training.train_passes_per_sample`）**：在一次rollout得到的样本上重复训练多轮。每一轮会按照`updates_per_rollout`将样本划分成若干更新批次，确保每个样本在一轮中只被使用一次。
- **定位比率裁剪调试（`training.debug_ratio`与`training.debug_ratio_dir`）**：开启`debug_ratio`后，会在`debug_ratio_dir`下按训练步保存被裁剪的`locate_ratio`调试信息，每个rank以`step_x/<rank>.json`形式记录最正向和最反向（各1000条，如不足则全量）的logits差值样本。