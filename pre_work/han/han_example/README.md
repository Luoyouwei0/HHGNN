# Heterogeneous Graph Attention Network (HAN) with DGL

This is an attempt to implement HAN with DGL's latest APIs for heterogeneous graphs.
The authors' implementation can be found [here](https://github.com/Jhy1993/HAN).

## Usage

`python main.py` for reproducing HAN's work on their dataset.

`python main.py --hetero` for reproducing HAN's work on DGL's own dataset from
[here](https://github.com/Jhy1993/HAN/tree/master/data/acm).  The dataset is noisy
because there are same author occurring multiple times as different nodes.

For sampling-based training, `python train_sampling.py`

## Performance

Reference performance numbers for the ACM dataset:

|                     | micro f1 score | macro f1 score |
| ------------------- | -------------- | -------------- |
| Paper               | 89.22          | 89.40          |
| DGL                 | 88.99          | 89.02          |
| Softmax regression (own dataset) | 89.66  | 89.62     |
| DGL (own dataset)   | 91.51          | 91.66          |

We ran a softmax regression to check the easiness of our own dataset.  HAN did show some improvements.



          
# 基于DGL的异构图注意力网络 (HAN)

这是一次尝试使用DGL最新的异构图API实现HAN的实践。原作者的实现可以在[这里](https://github.com/Jhy1993/HAN)找到。

## 使用方法

运行 `python main.py` 以复现HAN在其原始数据集上的实验结果。

运行 `python main.py --hetero` 以复现HAN在DGL自有数据集上的实验结果，数据集可从[这里](https://github.com/Jhy1993/HAN/tree/master/data/acm)获取。该数据集存在噪声，因为有相同的作者以不同节点的形式多次出现。

若要进行基于采样的训练，请运行 `python train_sampling.py`。

## 性能表现

ACM数据集的参考性能指标：

|                     | 微观F1分数 | 宏观F1分数 |
| ------------------- | -------------- | -------------- |
| 论文结果               | 89.22          | 89.40          |
| DGL实现结果                 | 88.99          | 89.02          |
| 软最大化回归（自有数据集） | 89.66  | 89.62     |
| DGL（自有数据集）   | 91.51          | 91.66          |

我们运行了软最大化回归来验证自有数据集的难易程度。HAN确实展现出了一定的性能提升。 

        