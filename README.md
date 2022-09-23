### Implementation of Neural Process(NP) and its Varaints
---

Neural Processes(NPs) combine the strengths of neural networks and Gaussian processes to achieve both flexible learning and fast prediction in stochastic processes. There are an extensive research to overcome issues from NP such as underfitting to context data and scalability to high dimensions. I'll introduce some important papers in NP family and provide implementations so that researchers who are new to NP can easily access how NP and its variants works.

|Model Name|Paper|Key Idea|Implementation|
|---|---|---|---|
|NP|[Neural Processes](https://arxiv.org/abs/1807.01622)|Combine GP and NN to achieve both flexibilty and fast prediction|Completed|
|ANP|[Attentive Neural Processes](https://arxiv.org/abs/1901.05761)|Insert attention module to prevent underfiitng to context data||
|BNP|[Bootstrapping Neural Processes](https://proceedings.neurips.cc/paper/2020/hash/492114f6915a69aa3dd005aa4233ef51-Abstract.html)|Improve robustness of NP using Bootstrapping||
