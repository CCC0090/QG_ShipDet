<div align="center">

<h1>QG-ShipDet</h1>

<p><b>面向 SAR 图像微小舰船检测的量子引导关系表示学习</b></p>

<p>Chengcheng Chen<sup>1</sup>, Yunyu Chen<sup>1</sup>, Jiaying Li<sup>2</sup>, Hongyu Chen<sup>3</sup>, Yueyang Li<sup>4</sup>,<br>
Yuhao Zhou<sup>5,&#42;</sup>, Nizhuan Wang<sup>4</sup>, Zhenxue He<sup>6</sup>, Xiang Wang<sup>7</sup></p>

<p><sub><sup>1</sup>广州航海学院 · <sup>2</sup>卡内基梅隆大学 · <sup>3</sup>同济大学 · <sup>4</sup>香港理工大学 · <sup>5</sup>临沂大学 · <sup>6</sup>河北农业大学 · <sup>7</sup>北京航空航天大学</sub>
<br><sub>&#42; 通讯作者</sub></p>

<p>
  <img alt="Paper" src="https://img.shields.io/badge/%E8%AE%BA%E6%96%87-IEEE%20TMM%20%C2%B7%20%E5%AE%A1%E7%A8%BF%E4%B8%AD-0B2545?style=flat-square">
  <img alt="Code" src="https://img.shields.io/badge/%E4%BB%A3%E7%A0%81-%E5%BD%95%E7%94%A8%E5%90%8E%E5%BC%80%E6%BA%90-C97A1B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%2011.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Datasets" src="https://img.shields.io/badge/%E6%95%B0%E6%8D%AE%E9%9B%86-HRSID%20%7C%20SSDD-0F7B84?style=flat-square">
</p>

<p>
  <a href="#方法概述">方法概述</a> ·
  <a href="README.md">English</a>
</p>

</div>

---

## 方法概述

SAR 具备全天时、全天候的海上监视能力，但复杂场景中的微小舰船仍然难以检测。困难不仅来自尺度：经过多级特征
下采样后，一艘小目标可能只剩极少数有效响应，而斑点噪声、海岸线、港口与人造结构会产生强度相当的竞争性散射
响应。此时实例级证据本身微弱且不可靠，单纯强化单个目标的特征表达收益有限。

**QG-ShipDet 将 SAR 微小舰船检测重新表述为「关系正则化的表示学习」。** 它不要求单个实例更具判别力，而是对
*实例之间的关系结构*施加监督：舰船与舰船之间的关系约束类内一致性，舰船与强散射难例杂波之间的关系约束可分性。
该监督由一个面向 SAR 的**量子关系教师**提供——*SAR 先验引导的量子态编码*（SPQSE）将多尺度语义、局部散射
统计与几何属性映射为目标级量子态；*微小舰船尺度感知量子关系引导*（TS-QRG）通过数据重上传与中间纠缠构建与
实例相关的保真度关系，并以尺度、难度、置信度三重加权的方式蒸馏到经典检测器。

量子教师仅在**训练阶段**存在。优化结束后，量子分支、关系损失与特征钩子全部移除，部署模型保持原有的推理拓扑、
参数量与 FLOPs——关系结构改变的是训练方式，而不是运行方式。在 HRSID 与 SSDD 基准上，QG-ShipDet 的
`mAP50:95` 分别提升 2.42 与 5.17 个百分点，微小目标（A = w·h < 32² pixels²）的 `AP_T` 分别提升 6.09 与 7.64
个百分点；在目标召回 `R = 0.90` 的工作点下，每幅图像的杂波虚警由 0.404 降至 0.267、由 0.302 降至 0.173。

论文目前在 *IEEE Transactions on Multimedia* 审稿中。如论文所述，代码、配置文件与预训练权重将在**录用后**于
本仓库公开，欢迎 watch / star 本仓库以获取通知。

---

**联系方式** — Chengcheng Chen（chenchengcheng@gzmtu.edu.cn）· 通讯作者 Yuhao Zhou
（zhouyuhao@tongji.edu.cn）。本工作受国家自然科学基金（No. 52602592）资助。
