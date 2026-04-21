# SALT-Track Command Guide

这份文档只保留后续正式实验要用的四个统一命名。
历史 yaml、checkpoint、结果和日志都已归档。

归档位置：
- yaml: `/root/user-data/wyp/ATCTrack_align/experiments/atctrack/archive_20260418/`
- output: `/root/user-data/wyp/ATCTrack_align/output/archive_20260418/`
- old cmd: `/root/user-data/wyp/ATCTrack_align/scripts/cmd_archive_20260418.md`

默认工作目录：

```bash
cd /root/user-data/wyp/ATCTrack_align
```

默认 Python：

```bash
/root/user-data/envs/wyp_vlt/bin/python
```

建议先激活环境：

```bash
cd /root/user-data/wyp/ATCTrack_align
source /root/user-data/envs/wyp_vlt/bin/activate
```

## 1. 四个正式实验命名

后续正式实验只保留四类：

1. `atctrack_base_semantic_full`
   主实验，视觉监督 + 文本监督，full finetune
2. `atctrack_base_semantic_visual`
   纯视觉监督对比
3. `atctrack_base_semantic_text`
   纯文本监督对比
4. `atctrack_base_semantic_lora`
   LoRA 训练方式对比

规则：
- 名字里不再写 `h100`、`trial`、`bigbs`、`short` 等超参信息。
- 超参差异写在 yaml 内容里，不写在名字里。
- 后续消融实验再追加清晰后缀，例如：
  - `atctrack_base_semantic_full_ablation_gate`
  - `atctrack_base_semantic_full_ablation_dataset`

## 2. 当前正式 yaml 文件

```text
experiments/atctrack/atctrack_base_semantic_full.yaml
experiments/atctrack/atctrack_base_semantic_visual.yaml
experiments/atctrack/atctrack_base_semantic_text.yaml
experiments/atctrack/atctrack_base_semantic_lora.yaml
```

当前四个正式 yaml 的统一原则：
- 相同训练集：`LASOT + VastTrack + TNL2K_train + OTB99_train`
- 相同基础 recipe：
  - `batch size = 25`
  - `epoch = 100`
  - `lr drop epoch = 80`
  - `save interval = 10`
  - `lr = 8e-5`
  - `sigmoid gate`
  - 语义监督采用两阶段权重调度：前 `40%` epoch 使用 `0.3`，后 `60%` epoch 使用 `0.1`
- 差异只保留在监督形式 / 训练方式本身

## 3. 通用环境变量

训练/推理都建议固定 CPU 线程：

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
```

训练常用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

推理常用：

```bash
export CUDA_VISIBLE_DEVICES=1
```

## 4. 四个正式实验训练命令

### 4.1 主实验

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python lib/train/run_training.py \
  --script atctrack \
  --config atctrack_base_semantic_full \
  --save_dir /root/user-data/wyp/ATCTrack_align/output \
  --use_lmdb 0 \
  --use_wandb 0
```

### 4.2 纯视觉对比

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python lib/train/run_training.py \
  --script atctrack \
  --config atctrack_base_semantic_visual \
  --save_dir /root/user-data/wyp/ATCTrack_align/output \
  --use_lmdb 0 \
  --use_wandb 0
```

### 4.3 纯文本对比

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python lib/train/run_training.py \
  --script atctrack \
  --config atctrack_base_semantic_text \
  --save_dir /root/user-data/wyp/ATCTrack_align/output \
  --use_lmdb 0 \
  --use_wandb 0
```

### 4.4 LoRA 对比

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python lib/train/run_training.py \
  --script atctrack \
  --config atctrack_base_semantic_lora \
  --save_dir /root/user-data/wyp/ATCTrack_align/output \
  --use_lmdb 0 \
  --use_wandb 0
```

## 5. 四个正式实验推理命令

通用模板：

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python tracking/test.py \
  --tracker_name atctrack \
  --tracker_param <tracker_param> \
  --dataset_name tnl2k \
  --threads 2 \
  --num_gpus 1 \
  --ckpt_path <checkpoint_path>
```

### 5.1 主实验

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python tracking/test.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_full \
  --dataset_name tnl2k \
  --threads 12 \
  --num_gpus 1 \
  --ckpt_path /root/user-data/wyp/ATCTrack_align/output/checkpoints/train/atctrack/atctrack_base_semantic_full/ATCTrack_ep0100.pth.tar
```

### 5.2 纯视觉对比

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python tracking/test.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_visual \
  --dataset_name tnl2k \
  --threads 2 \
  --num_gpus 1 \
  --ckpt_path /root/user-data/wyp/ATCTrack_align/output/checkpoints/train/atctrack/atctrack_base_semantic_visual/ATCTrack_ep0100.pth.tar
```

### 5.3 纯文本对比

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python tracking/test.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_text \
  --dataset_name tnl2k \
  --threads 8 \
  --num_gpus 1 \
  --ckpt_path /root/user-data/wyp/ATCTrack_align/output/checkpoints/train/atctrack/atctrack_base_semantic_text/ATCTrack_ep0100.pth.tar
```

### 5.4 LoRA 对比

```bash
cd /root/user-data/wyp/ATCTrack_align
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
/root/user-data/envs/wyp_vlt/bin/python tracking/test.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_lora \
  --dataset_name tnl2k \
  --threads 2 \
  --num_gpus 1 \
  --ckpt_path /root/user-data/wyp/ATCTrack_align/output/checkpoints/train/atctrack/atctrack_base_semantic_lora/ATCTrack_ep0100.pth.tar
```

## 6. 四个正式实验评测命令

### 6.1 主实验

```bash
cd /root/user-data/wyp/ATCTrack_align
/root/user-data/envs/wyp_vlt/bin/python tracking/analysis_results.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_full \
  --dataset_name tnl2k
```

### 6.2 纯视觉对比

```bash
cd /root/user-data/wyp/ATCTrack_align
/root/user-data/envs/wyp_vlt/bin/python tracking/analysis_results.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_visual \
  --dataset_name tnl2k
```

### 6.3 纯文本对比

```bash
cd /root/user-data/wyp/ATCTrack_align
/root/user-data/envs/wyp_vlt/bin/python tracking/analysis_results.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_text \
  --dataset_name tnl2k
```

### 6.4 LoRA 对比

```bash
cd /root/user-data/wyp/ATCTrack_align
/root/user-data/envs/wyp_vlt/bin/python tracking/analysis_results.py \
  --tracker_name atctrack \
  --tracker_param atctrack_base_semantic_lora \
  --dataset_name tnl2k
```

## 7. 当前状态说明

当前正式配置统一为 100 epoch，checkpoint 每 10 轮保存一次。
现在已经完成的老结果全部被归档，不再作为正式命名主线继续扩展。
后续新的正式实验、checkpoint、结果目录、日志目录都统一使用这四个名字。



cd /root/user-data/wyp/ATCTrack_align
/root/user-data/envs/wyp_vlt/bin/python -m tensorboard.main \
  --logdir /root/user-data/wyp/ATCTrack_align/tensorboard \
  --port 6006 \
  --host 0.0.0.0