# SV-Softmax
Implementation for paper SV-Softmax: Large-Margin Softmax Loss using Synthetic Virtual Class.

In our paper, we introduce a margin adaptive synthetic virtual Softmax loss with virtual prototype insertion strategy, which emphasize the importance of misclassified hard samples.

The proposed unified virtual class loss framework is:

$$
\begin{align}
L &= \frac{1}{N}\sum_{i=1}^{N} L_{i}
 =-\frac{1}{N}\sum_{i=1}^{N} \log\frac{e^{w_{y_{i}}^{T}z_{i}}}{\sum_{j=1}^{C}e^{w_{j}^{T}z_{i}}+e^{w_{v}^{T}z_{i}}}, \\
w_{v} &= 
\begin{cases}
    w_{synth} = \frac{\|w_{y_i}\|h}{\|h\|}, &\text{if } \  w_{y_i} z_i \geq \max_{j \neq y_i} w_j z_i \\
    w_{virt} = \frac{\|w_{y_i}\|z_i}{\|z_i\|}, &\text{if } \ w_{y_i}z_i < \max_{j \neq y_i} w_j z_i \\
\end{cases} ,\\
h &= m\frac{z_i}{\|z_i\|} - (1-m)\frac{w_{y_i}}{\|w_{y_i}\|}, \quad m\in[0,1].
\end{align}
$$


**The full experiment code is being cleaned up and will release soon.** 
