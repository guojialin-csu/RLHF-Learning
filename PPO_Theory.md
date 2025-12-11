#### 理论推导
1. PPO的目标是：最大化在分布 $P_\theta$ 中采样的轨迹 $\tau$ 的总回报 $R(\tau)$  
${E(R(\tau))}_ {\tau\sim P_\theta(\tau)}=\sum_\tau P_\theta(\tau)R(\tau)$  

2. 转换成目标函数的梯度  
$\nabla{E(R(\tau))}_ {\tau\sim P_\theta(\tau)}=\sum_\tau \nabla P_\theta(\tau)R(\tau)$  

3. 转换为可采样的形式  
$\nabla{E(R(\tau))}_{\tau\sim P_\theta(\tau)}=\sum_\tau \nabla P_\theta(\tau)R(\tau) \frac{P_\theta(\tau)}{P_\theta(\tau)}$  
$=\sum_\tau P_\theta(\tau)R(\tau) \frac{\nabla P_\theta(\tau)}{P_\theta(\tau)}$  
$=\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \frac{\nabla P_\theta(\tau^{n})}{P_\theta(\tau^{n})}$  
$=\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \nabla \log P_\theta(\tau^{n})$  

4. 轨迹的概率是轨迹上的所有动作的概率连乘得到  
$\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \nabla \log P_\theta(\tau^{n})=\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \nabla \log \prod_{t=1}^{T_n} P_\theta(a_t^n | s_t^n)$  
$=\frac{1}{N} \sum_{n=1}^{N} R(\tau^{n}) \nabla \sum_{t=1}^{T_n} \log P_\theta(a_t^n | s_t^n)$  
$=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^{n}) \nabla \log P_\theta(a_t^n | s_t^n)$  

5. 根据马尔可夫决策的特性进行修正:后面的动作不会影响前面的动作，并且当前动作的影响会随着时间的推移逐渐衰减。因此，将整个轨迹的回报更替为**由当前时刻到结束奖励的总和**  
$R(\tau^{n}) \rightarrow \sum_{t'=t}^{T_n} \gamma^{t'-t} r_t^n = R_t^n$  

6. 对选择每个动作的奖励进行归一化（减去在状态 $s_t^n$ 下选择任意动作的平均价值 $B(s_t^n)$ ），期望好的动作奖励大于0，坏的动作奖励小于0，突出不同动作的价值差异有利于模型收敛  
$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R_t^n-B(s_t^n)) \nabla \log P_\theta(a_t^n | s_t^n)$  

7. $R_t^n$ 是根据采样得到的，训练不稳定，转化为期望形式 $Q_\theta(a_t^n, s_t^n)$ ，即动作价值函数
$R_t^n \rightarrow Q_\theta(a_t^n, s_t^n)$  

8. 同样地，将 $B(s_t^n)$ 转化为状态价值函数 $V_\theta(s_t^n)$ ， $Q_\theta(a_t^n, s_t^n)$ 与 $V_\theta(s_t^n)$ 相减代表动作 $a_t^n$ 在状态 $s_t^n$ 下的优势  
$A_\theta(a_t^n, s_t^n)=Q_\theta(a_t^n, s_t^n)-V_\theta(s_t^n)$  

9. 使用GAE优势函数，平衡采样的偏差和方差  
$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_\theta^{GAE}(a_t^n, s_t^n) \nabla \log P_\theta(a_t^n | s_t^n)$  

10. On Policy转换为Off Policy(使用参考策略 $\theta'$ 进行数据采样，来更新策略 $\theta$ )  
$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_\theta^{GAE}(a_t^n, s_t^n) \nabla \log P_\theta(a_t^n | s_t^n)$  
$= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(a_t^n, s_t^n) \frac{P_\theta(a_t^n | s_t^n)}{P_{\theta'}(a_t^n | s_t^n)} \nabla \log P_\theta(a_t^n | s_t^n)$  
$= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(a_t^n, s_t^n) \frac{\nabla P_\theta(a_t^n | s_t^n)}{P_{\theta'}(a_t^n | s_t^n)}$  

11. Policy Loss的最终表达形式  
$loss = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(a_t^n, s_t^n) \frac{P_\theta(a_t^n | s_t^n)}{P_{\theta'}(a_t^n | s_t^n)}$  

12. 避免策略更新与参考策略相差太大，在最终loss中增加当前策略与参考策略的KL散度  
$loss = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(a_t^n, s_t^n) \frac{P_\theta(a_t^n | s_t^n)}{P_{\theta'}(a_t^n | s_t^n)}+\beta KL(P_\theta, P_{\theta'})$  

13. 或者通过截断来约束策略更新的步长  
$loss = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \min (A_{\theta'}^{GAE}(a_t^n, s_t^n) \frac{P_\theta(a_t^n | s_t^n)}{P_{\theta'}(a_t^n | s_t^n)}, clip(\frac{P_\theta(a_t^n | s_t^n)}{P_{\theta'}(a_t^n | s_t^n)},1-\epsilon, 1+\epsilon)A_{\theta'}^{GAE}(a_t^n, s_t^n))$
