# CGIL: Intervention-based Counterfactual for Generalizable Imitation Learning

该项目目的是提高模仿学习的分布外泛化性能，避免模型在遇到陌生场景时分布外泛化性能低而引起智能体的错误累积。

## 目录
- [安装](#安装)
- [使用说明](#使用说明)
- [功能清单](#功能清单)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 安装
1. 克隆仓库： 
   ```bash
   git clone https://github.com/betabeta123/causal_il.git

2. 安装依赖
## 使用说明

caufea_mining：包含因果发现算法（PC和FCI算法）以及互信息方法；

diff_distrib：模拟了7种不同数据分布；

counfac_aug：因果方程的拟合、基于干预的反事实数据增强、增强后的数据集合并；

IL_model、IL1、IL2：用到的模仿学习模型；

data：实验所需数据

因果关系发现：数据挖掘算法包

## 功能清单
- 功能一：从观测数据中找到因果关系并进行确认；
- 功能二：基于因果关系拟合因果方程；
- 功能三：基于干预的反事实数据增强。

![aaaaa](https://github.com/user-attachments/assets/631a9751-dad3-4065-8368-40e99cd89f53)

## 许可证
本项目采用 MIT 许可证。详细信息请参阅 [LICENSE](LICENSE) 文件。

## 联系方式
如有疑问，请联系:lltian@stu.ecnu.edu.cn

