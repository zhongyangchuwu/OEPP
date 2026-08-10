import copy

import torch

from oepp.models.helpers import AverageMeter
from oepp.training.pdpp_runtime import unwrap_model

from .accuracy import accuracy


def cycle(dl):
    while True:
        yield from dl


def validate_pdpp_loss_weights(ce_weight: float, mse_weight: float) -> None:
    if ce_weight < 0 or mse_weight < 0:
        raise ValueError("PDPP CE and MSE weights must be non-negative")
    if ce_weight == 0 and mse_weight == 0:
        raise ValueError("PDPP requires at least one active CE or MSE loss branch")


def pdpp_action_objective(
    predicted: torch.Tensor,
    labels: torch.Tensor,
    train_text_tensor: torch.Tensor,
    *,
    ce_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    """Build the legacy normalized PDPP action objective and auditable components."""
    validate_pdpp_loss_weights(ce_weight, mse_weight)
    if predicted.ndim != 3 or labels.shape != predicted.shape[:2]:
        raise ValueError("PDPP predictions and labels must have shapes [B,T,D] and [B,T]")

    batch_size, horizon, _ = predicted.shape
    objective = predicted.new_zeros(())
    ce_total = predicted.new_zeros(())
    mse_total = predicted.new_zeros(())
    epsilon = torch.finfo(predicted.dtype).eps

    for step in range(horizon):
        step_prediction = predicted[:, step, :]
        step_labels = labels[:, step]

        if mse_weight > 0:
            ground_truth = train_text_tensor[step_labels]
            step_mse = (
                torch.nn.functional.mse_loss(step_prediction, ground_truth, reduction="sum")
                / batch_size
            )
            mse_total = mse_total + step_mse
            objective = objective + mse_weight * step_mse / step_mse.detach().clamp_min(epsilon)

        if ce_weight > 0:
            similarity = torch.nn.functional.cosine_similarity(
                step_prediction.unsqueeze(1), train_text_tensor.unsqueeze(0), dim=2
            )
            legacy_probabilities = torch.nn.functional.softmax(similarity / 0.1, dim=1)
            step_ce = torch.nn.functional.cross_entropy(legacy_probabilities, step_labels)
            ce_total = ce_total + step_ce
            objective = objective + ce_weight * step_ce / step_ce.detach().clamp_min(epsilon)

    metrics = {
        "objective": float(objective.detach().item() / horizon),
        "ce": float(ce_total.detach().item() / horizon) if ce_weight > 0 else None,
        "mse": float(mse_total.detach().item() / horizon) if mse_weight > 0 else None,
    }
    return objective, metrics


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:
    def __init__(
        self,
        diffusion_model,
        datasetloader1,
        datasetloader2,
        datasetloader3,
        datasetloader4,
        ema_decay=0.995,
        train_lr=1e-5,
        gradient_accumulate_every=1,
        step_start_ema=400,
        update_ema_every=10,
        log_freq=100,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataloader1 = cycle(datasetloader1)
        self.dataloader2 = cycle(datasetloader2)
        self.dataloader3 = cycle(datasetloader3)
        self.dataloader4 = cycle(datasetloader4)
        self.optimizer = torch.optim.AdamW(
            diffusion_model.parameters(), lr=train_lr, weight_decay=0.0
        )

        self.reset_parameters()
        self.step = 0
        self.last_loss_metrics: dict[str, float | None] = {
            "objective": 0.0,
            "ce": None,
            "mse": None,
        }

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps, if_calculate_acc, args, scheduler, train_text_tensor):
        validate_pdpp_loss_weights(float(args.para_ce), float(args.para_mse))
        self.model.train()
        self.ema_model.train()
        losses1 = AverageMeter()
        ce_losses = AverageMeter()
        mse_losses = AverageMeter()
        self.optimizer.zero_grad()

        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch1 = next(self.dataloader1)
                _, start, end, _, _, _, _, labels, _ = batch1
                bs1, T1 = batch1[-2].shape  # [bs, (T+1), ob_dim]
                labels = labels.cuda()
                img_tensors1 = torch.zeros(
                    (
                        bs1,
                        T1,
                        args.class_dim + args.action_dim + args.observation_dim + args.horizon_dim,
                    )
                )
                img_tensors1[:, 0, args.horizon_dim + args.class_dim + args.action_dim :] = (
                    start.cuda().contiguous().float()
                )
                img_tensors1[:, -1, args.horizon_dim + args.class_dim + args.action_dim :] = (
                    end.cuda().contiguous().float()
                )
                img_tensors1 = img_tensors1.cuda()

                if args.class_dim != 0:
                    assert 0
                else:
                    cond1 = {
                        0: img_tensors1[
                            :, 0, args.horizon_dim + args.class_dim + args.action_dim :
                        ].float(),
                        T1 - 1: img_tensors1[
                            :, -1, args.horizon_dim + args.class_dim + args.action_dim :
                        ].float(),
                    }

                if args.horizon_dim != 0:
                    horizon_onehot1 = torch.zeros((bs1, args.horizon_dim))
                    horizon_onehot1[:, 0] = 1.0
                    temp1 = horizon_onehot1.unsqueeze(1)
                    horizon_onehot1 = temp1.repeat(1, T1, 1)
                    img_tensors1[:, :, : args.horizon_dim] = horizon_onehot1
                    cond1["horizon"] = horizon_onehot1

                x1 = img_tensors1.float()
                x_output = unwrap_model(self.model).loss(x1, cond1)
                x_output = x_output[
                    :,
                    :,
                    args.horizon_dim + args.class_dim : args.horizon_dim
                    + args.class_dim
                    + args.action_dim,
                ]
                loss, loss_metrics = pdpp_action_objective(
                    x_output,
                    labels,
                    train_text_tensor,
                    ce_weight=float(args.para_ce),
                    mse_weight=float(args.para_mse),
                )

                loss = loss / self.gradient_accumulate_every
                loss.backward()
                losses1.update(loss_metrics["objective"], bs1)
                if loss_metrics["ce"] is not None:
                    ce_losses.update(loss_metrics["ce"], bs1)
                if loss_metrics["mse"] is not None:
                    mse_losses.update(loss_metrics["mse"], bs1)
                self.optimizer.step()
                self.optimizer.zero_grad()

            scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.step += 1

        self.last_loss_metrics = {
            "objective": float(losses1.avg),
            "ce": float(ce_losses.avg) if ce_losses.count else None,
            "mse": float(mse_losses.avg) if mse_losses.count else None,
        }

        if if_calculate_acc:
            with torch.no_grad():
                x_output = self.ema_model(cond1, T1, if_jump=True)
                x_output = x_output[
                    :,
                    :,
                    args.horizon_dim + args.class_dim : args.horizon_dim
                    + args.class_dim
                    + args.action_dim,
                ]

                pred_logits = None

                for j in range(T1):
                    frames_embedding = x_output[:, j, :]
                    sim_logits = torch.nn.functional.cosine_similarity(
                        frames_embedding.unsqueeze(1), train_text_tensor.unsqueeze(0), dim=2
                    )  # 256 666
                    sim_logits = sim_logits / 0.1
                    sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
                    sim_logits_softmax = sim_logits_softmax.unsqueeze(1)
                    # print(sim_logits_softmax)
                    if pred_logits is None:
                        pred_logits = sim_logits_softmax
                    else:
                        pred_logits = torch.cat((pred_logits, sim_logits_softmax), dim=1)

                pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
                (acc11, acc51), trajectory_success_rate1, MIoU11, MIoU21, a0_acc1, aT_acc1 = (
                    accuracy(pred_logits.cpu(), labels.view(-1).cpu(), topk=(1, 5), max_traj_len=T1)
                )

                return (
                    torch.tensor(losses1.avg),
                    acc11,
                    acc51,
                    torch.tensor(trajectory_success_rate1),
                    torch.tensor(MIoU11),
                    torch.tensor(MIoU21),
                    a0_acc1,
                    aT_acc1,
                )

        else:
            return torch.tensor(losses1.avg)
