<div align="center">

<h1>QG-ShipDet</h1>

<p><b>Quantum-Guided Relational Representation Learning for Tiny Ship Detection in SAR Imagery</b></p>

<p>Chengcheng Chen<sup>1</sup>, Yunyu Chen<sup>1</sup>, Jiaying Li<sup>2</sup>, Hongyu Chen<sup>3</sup>, Yueyang Li<sup>4</sup>,<br>
Yuhao Zhou<sup>5,&#42;</sup>, Nizhuan Wang<sup>4</sup>, Zhenxue He<sup>6</sup>, Xiang Wang<sup>7</sup></p>

<p><sub><sup>1</sup>Guangzhou Maritime University · <sup>2</sup>Carnegie Mellon University · <sup>3</sup>Tongji University · <sup>4</sup>The Hong Kong Polytechnic University · <sup>5</sup>Linyi University · <sup>6</sup>Hebei Agricultural University · <sup>7</sup>Beihang University</sub>
<br><sub>&#42; Corresponding author</sub></p>

<p>
  <img alt="Paper" src="https://img.shields.io/badge/Paper-IEEE%20TMM%20(under%20review)-0B2545?style=flat-square">
  <img alt="Code" src="https://img.shields.io/badge/Code-released%20upon%20acceptance-C97A1B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%2011.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Datasets" src="https://img.shields.io/badge/Datasets-HRSID%20%7C%20SSDD-0F7B84?style=flat-square">
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="README.zh-CN.md">中文说明</a>
</p>

</div>

---

## Overview

SAR imaging supports all-day and all-weather maritime surveillance, yet tiny ships remain difficult to detect in
complex scenes. The difficulty is not only scale: after repeated feature downsampling, a tiny ship may survive as
only a handful of effective responses, while speckle, coastlines, ports and man-made structures produce competing
scattering patterns of comparable strength. In this regime instance-level evidence is weak and unreliable, so
strengthening the representation of a single target yields limited returns.

**QG-ShipDet recasts SAR tiny-ship detection as relation-regularized representation learning.** Instead of asking
one instance to be more discriminative, it supervises the *structure between instances*: ship–ship pairs encode
class consistency, while ship–hard-clutter pairs encode separability from confusable strong-scattering background
regions. The supervision is supplied by an SAR-aware **quantum relational teacher**. *SAR Prior-Guided Quantum
State Encoding* (SPQSE) maps multiscale semantics, local scattering statistics and geometry into object-level
quantum states; *Tiny-Ship Scale-Aware Quantum Relation Guidance* (TS-QRG) constructs data-dependent fidelity
relations through data re-uploading and intermediate entanglement, and distills them into the classical detector
with scale-, hardness- and confidence-aware weighting.

The teacher is active **only during training**. After optimization, the quantum branch, the relation losses and the
feature hooks are removed, so the deployed detector keeps its original inference topology, parameter count and
FLOPs — relation structure changes how the detector is trained, not how it runs. On the HRSID and SSDD benchmarks,
QG-ShipDet improves `mAP50:95` by 2.42 and 5.17 points, and `AP_T` on tiny ships (A = w·h < 32² pixels²) by 6.09
and 7.64 points, while reducing clutter false alarms per image at the target-recall operating point `R = 0.90`
from 0.404 to 0.267 and from 0.302 to 0.173, respectively.

The manuscript is currently under review at *IEEE Transactions on Multimedia*. As stated in the paper, the code,
configuration files and pretrained weights will be made publicly available in this repository **upon acceptance**;
please watch or star the repository to be notified.

---

**Contact** — Chengcheng Chen (chenchengcheng@gzmtu.edu.cn) · Yuhao Zhou, corresponding author
(zhouyuhao@tongji.edu.cn). This work was supported by the National Natural Science Foundation of China
(No. 52602592).
