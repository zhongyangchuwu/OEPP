# Open-Event Procedure Planning in Instructional Videos

*[Yilu WU]*<sup>1</sup>, 
*[Hanlin Wang]*<sup>1</sup>, 
*[Jing Wang]*<sup>1</sup>, 
*[Limin Wang]*<sup>1</sup>

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University

<p align = "center"> 
<img src="img\\intro-sample.png"  width="600" />
</p>

**Abstract** : Given the current visual observations, the traditional procedure planning task in instructional videos requires a model to generate goal-directed plans within a given action space. 
All previous methods for this task conduct training and inference under the same action space, and they can only plan for pre-defined events in the training set. We argue this setting is not applicable for human assistance in real lives and aim to propose a more general and practical planning paradigm. 
Specifically, in this paper, we introduce a new task named Open-event Procedure Planning (OEPP), which extends the traditional procedure planning to the open-event setting. OEPP aims to verify whether a planner can transfer the learned knowledge to similar events that have not been seen during training. 
We rebuild a new benchmark of OpenEvent for this task based on existing datasets and divide the events involved into base and novel parts. During the data collection process, we carefully ensure the transfer ability of procedural knowledge for base and novel events by evaluating the similarity between the descriptions of different event steps with multiple stages. 
Based on the collected data, we further propose a simple and general framework specifically designed for OEPP, and conduct extensive study with various baseline methods, providing a detailed and insightful analysis on the results for this task.

## Dataset
Our data splits and annotations are under `data`.

| **File**                             | **Description**    |
|--------------------------------------|--------------------|
| data/train_train_base_dataset_1.json | train dataset      |
| data/train_train_val_dataset_1.json  | val dataset        |
| data/novel_dataset_1.json            | test novel dataset |
| data/test_base_dataset_1.json        | test base dataset  |
| data/base_action_pool_1.json         | base action pool   |
| data/novel_action_pool_1.json        | novel action pool  |
| data/total_action_pool_1.json        | total action pool  |
| data/task_info.json                  | event info         |

## Install Dependecny

`conda create --name oepp python=3.9`

`conda activate oepp`

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

`pip install -r requirements.txt`

## Download features

Download the features [link](https://drive.google.com/drive/folders/1IKrEnPhIvQhBN-tiIvtn_bG6EtDE8bNs?usp=drive_link)
and modify the path to the features in the `dataset/dataset.py Line51-62`.
The S3D feature is extracted by [P3IV](https://github.com/JoeHEZHAO/procedure-planing).
Note that our videoclip features are currently only available for OEPP. We will release the complete features as soon as possible.

## Training

### MLP-based 
You can run the code using the following command, and the results will be in `results/MLP`.

`python train.py --config=MLP_config.yaml`

### Transformer-based 

You can run the code using the following command, and the results will be in `results/attention`.

`python train.py --config=attention_config.yaml`

### PDPP legacy path

The original PDPP command below is retained in Git history only; it can resume a placeholder directory and previously evaluated test sets during training. Use the fresh Experiment 4 PDPP command below instead, which creates state-dict checkpoints and reserves Base/Novel evaluation for the exporter.

## Experiment 4: fresh embedding analysis

The legacy `train.py` and `eval.py` remain available for reproducing their original discrete metrics. For embedding analysis, train from fresh initialization with the state-dict checkpoint runner; it supports both the MLP and Transformer OEPP baselines and never selects on Base or Novel test metrics.

Run this before calibration or training. It verifies every annotation/action against its split pool, all expected feature file paths, action embeddings, and CUDA/Torch; it exits nonzero on any failure.

```bash
python embedding_preflight.py \
  --feature videoclip \
  --verify-feature-content \
  --output results/experiment4/preflight_videoclip.json
python -m unittest discover -s tests -v
```
Server acceptance criteria: preflight exits 0; unit tests pass; each direct run writes `last.pt`, `best.pt`, and `selection.json`; direct export writes Base `(1138, 3, 768)` and Novel `(1691, 3, 768)` raw tensors, 3,414 / 5,073 CSV step rows, `summary_metrics.json`, and all seven named figures. PDPP must meet the same export shape/count checks in its own output directory using the documented `--sampling_seed`.

```bash
# Timing calibration only: a separate 10-epoch fresh run.
CUDA_VISIBLE_DEVICES=0 python train_embeddings.py \
  --config attention_config.yaml \
  --epochs 10 \
  --run-dir results/experiment4/attention_t3_seed42_calibration

# Fresh 200-epoch Transformer run.
CUDA_VISIBLE_DEVICES=0 python train_embeddings.py \
  --config attention_config.yaml \
  --run-dir results/experiment4/attention_t3_seed42

# Fresh 200-epoch MLP comparator.
CUDA_VISIBLE_DEVICES=0 python train_embeddings.py \
  --config MLP_config.yaml \
  --run-dir results/experiment4/mlp_t3_seed42

# Export Base/Novel continuous embeddings and all pre-specified figures.
CUDA_VISIBLE_DEVICES=0 python export_embeddings.py \
  --checkpoint results/experiment4/attention_t3_seed42/best.pt \
  --output-dir embedding_results/attention_t3_seed42
```

PDPP is a separate stochastic model and must use a separate run/result directory. It now saves fresh `last.pt` and validation-selected `best.pt`; do not pass `--resume` for a new run and do not pass `--test_during_training`.

```bash
CUDA_VISIBLE_DEVICES=0 python pdpp_train.py \
  --gpu 0 \
  --checkpoint_root results/experiment4/pdpp_checkpoints \
  --checkpoint_dir pdpp_t3_seed217 \
  --log_root results/experiment4/pdpp_logs \
  --horizon 3 --feat videoclip --split 1 --is_pad 1 \
  --para_mse 0.2 --para_ce 1.0 --lr 0.0005 \
  --batch_size 32 --batch_size_val 32 --epochs 200 \
  --num_thread_reader 4 --pin_memory --evaluate --sampling_seed 42

CUDA_VISIBLE_DEVICES=0 python export_pdpp_embeddings.py \
  --checkpoint results/experiment4/pdpp_checkpoints/pdpp_t3_seed217/best.pt \
  --output-dir embedding_results/pdpp_t3_seed217 \
  --sampling_seed 42
```

Before a server run, verify CUDA/Torch, write access, the four annotation JSON files, and `/data0/wuyilu/data/OEPP_videoclip`. No dataset conversion is required: `Seq_action` reads the original annotations and precomputed feature files. The embedding path records stable source-window metadata and rejects an action absent from its selected pool. `matplotlib==3.8.3` is required for the figures.