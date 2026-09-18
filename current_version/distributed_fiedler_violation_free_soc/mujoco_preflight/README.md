# 四机 MuJoCo 预备实物实验：现场操作入口

本目录只保留一套实验：**四机、固定四环、5 Hz 高层控制、$K=2$、一次完整通信轮10 ms**。不要从其他脚本选择 profile 或扫描参数；当前唯一运行入口是 `run_current_onboard_preflight.py`。

## 1. 先看这六个数字

| 项目 | 固定值 |
|---|---:|
| 高层控制 | 5 Hz，200 ms/周期 |
| MuJoCo/速度内环 | 200 Hz，5 ms/步 |
| 每次有限共识 | $K=2$ |
| 完整邻居通信轮 | 10 ms/轮，按100 Hz服务能力建模 |
| 每控制周期 | 12轮 |
| 平均通信需求 | 60轮/s |

这里的10 ms是“节点上行—基站路由—邻居接收”的**完整一轮**，不是单个UDP包延迟。它目前是仿真假设，不是目标板和真实Wi-Fi的测量结果。

## 2. 直接运行

在上一级 `distributed_fiedler_violation_free_soc` 目录执行：

```powershell
python -m pip install -r requirements.txt
python -m mujoco_preflight.run_current_onboard_preflight `
  --outdir mujoco_preflight/outputs_current_reproduction
python -m pytest -q test_mujoco_preflight.py
```

输出目录必须不存在；程序会拒绝覆盖已有数据。

## 3. 当前冻结结果

唯一冻结结果目录：`outputs_current_onboard_k02_100hz_v1/`。

| 指标 | 结果 | 判读 |
|---|---:|---|
| 最小真实 $\lambda_2$ | 0.588418 | 高于0.50 |
| 阈值以下采样点 | 0/200 | 当前采样场景通过 |
| 最小精确命令屏障裕量 | 0.370730 | 正 |
| 最小实际速度屏障裕量 | 0.260354 | 正 |
| 最大子空间正弦误差 | 0.110568 | 约 $6.35^\circ$ |
| 最大节点Ritz绝对误差 | 0.308677 | 只可作粗诊断 |
| 最大投影周期耗时 | 约122.1 ms | 低于200 ms；也低于160 ms建议放飞门槛 |
| 最大命令生效延迟 | 125 ms | 向5 ms物理步长量化后 |
| deadline miss | 0/200 | 固定延迟模型下通过 |
| 最小机间距离 | 1.11847 m | 高于0.40 m预备线 |
| 最小围栏余量 | 0.64261 m | 正 |

结果文件只包括当前实验：

- `fig_current_onboard_candidate.png/.pdf`：通信、连通性、Ritz、屏障和时延；
- `fig_mujoco_preflight.png/.pdf`：轨迹、谱、子空间和周期；
- `preflight_config.json`：唯一配置与12轮通信分解；
- `preflight_summary.csv`：汇总指标；
- `preflight_timeseries.csv`：200周期逐周期记录；
- `mujoco_state_proposed.npz`：MuJoCo状态回放。

## 4. 每周期12轮从哪里来

| 阶段 | 轮数 |
|---|---:|
| estimator diffusion | 1 |
| mean consensus | 2 |
| Gram consensus | 2 |
| Procrustes consensus | 2 |
| 邻居模型包 | 1 |
| 节点Ritz诊断 | 2 |
| allocation resource | 1 |
| allocation dual residual | 1 |
| **总计** | **12** |

每个节点每周期求解一次局部SOCP，因此每节点5次/s，四节点合计20次/s。运动开始前还有40个固定状态 estimator warm-up step，每步7轮，共280个启动通信轮；实物设备必须在解锁前完成预热。

## 5. Ritz误差怎么处理

节点Ritz标量不进入局部SOCP，只用于记录。旧的“误差不超过0.08”不是安全定理或放飞证书。当前最大误差0.308677意味着 $K=2$ 下节点Ritz显示较粗，不能声称节点精确估计了 $\lambda_2$。

如果未来要用它触发在线安全逻辑，必须先建立单侧误差界 $\varepsilon_i$，并检查

$$\widehat\lambda_{2,i}-\varepsilon_i\ge\lambda_{\rm req}.$$

## 6. 实物人员下一步做什么

1. 在四块目标板和真实无线接入点上运行完整12轮消息序列。
2. 记录 release-to-command 平均、p95、p99和最大延迟，以及丢包、重传、CPU、温度。
3. 连续压力测试要求零deadline miss；建议最大端到端延迟小于160 ms。
4. 先做Mocap回放和无桨HIL，再做单机、四机无扰动、最后完整40 s任务。
5. 逐项验证geofence、Mocap timeout、网络timeout、hover、land、kill和人工急停。

当前MuJoCo结果不能替代这些硬件测试，也不能证明随机丢包、异步通信或采样间连续时间安全。

## 7. 只需要认识这些文件

| 文件 | 用途 |
|---|---|
| `run_current_onboard_preflight.py` | 唯一实验入口、数据保存和汇总 |
| `scenario.py` | 唯一场景与固定参数 |
| `current_onboard_profile.py` | 12轮/周期、60轮/s核算 |
| `distributed_stack.py` | 分布式估计、allocation和局部SOCP |
| `network_emulator.py` | 四环邻居通信和固定逐轮延迟 |
| `mujoco_plant.py` | MuJoCo植物与200 Hz速度内环 |
| `plot_current_onboard_result.py` | 通信与闭环结果主图 |
| `plot_results.py` | 轨迹与通用诊断图 |
| `scene_four_uav.xml` | 四机MuJoCo场景 |
| `../test_mujoco_preflight.py` | 当前实验回归测试 |

更完整的上板、HIL和飞行放行流程见上一级 `../MuJoCo实验实物迁移方案.md`。
