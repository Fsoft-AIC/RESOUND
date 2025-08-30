# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import sys
from dataclasses import dataclass, field
import torch.nn.functional as F
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
logger = logging.getLogger(__name__)
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig
#logger = logging.getLogger(name)
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@dataclass
class MultiTargetCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    mel_weight: float = field(default=1., metadata={"help": "weight for mel loss"})
    f0_weight: float = field(default=1., metadata={"help": "weight for f0 loss"})
    energy_weight: float = field(default=1., metadata={"help": "weight for energy loss"})
    hubert_weight: float = field(default=1., metadata={"help": "weight for energy loss"})

@register_criterion("multi_target", dataclass=MultiTargetCriterionConfig)
class LabelSmoothedCrossEntropyCriterionLengthMatch(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        mel_weight,
        f0_weight,
        energy_weight,
        hubert_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.criterion_l1 = torch.nn.L1Loss(reduction='none')
        self.criterion_sc = SpectralConvergenceLoss()

        self.criterion_l1_f0 = torch.nn.L1Loss(reduction='none')
        self.criterion_l1_energy = torch.nn.L1Loss(reduction='none')

        self.mel_weight = mel_weight
        self.hubert_weight = hubert_weight
        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.padding_idx = task.target_dictionary.pad()
        self.padding_idx_f0 = task.target_f0_dictionary.pad()
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        hubert_loss = loss * self.hubert_weight
        loss = hubert_loss
       # loss = 0
        
        if net_output["encoder_out_mel"] is not None:
            pred, targ = net_output["encoder_out_mel"], sample["mel"]
            targ_mask = ~sample['net_input']['padding_mask'].repeat_interleave(4, dim=1)

            crop_len = min(targ_mask.sum(1).max().item(), pred.size(1), targ.size(1))

            pred = pred[:,:crop_len].contiguous()
            targ = targ[:,:crop_len].contiguous()
            targ_mask = targ_mask[:,:crop_len].contiguous()

            #assert 1==0, f"{(pred.shape)}, {targ.shape}, {targ_mask.shape}"
            pred_list, targ_list = [], []
            for p, t, m in zip(pred, targ, targ_mask):
                pred_list.append(p[m])
                targ_list.append(t[m])

            if self.sentence_avg:
                mel_loss = ((self.criterion_l1(pred, targ).mean(-1) * targ_mask).sum(1) / targ_mask.sum(1)).sum()
            else:
                mel_loss = (self.criterion_l1(pred, targ).mean(-1) * targ_mask).sum()

            sc_loss = self.criterion_sc(pred_list, targ_list, self.sentence_avg)

            mel_loss += sc_loss
            mel_loss = mel_loss * self.mel_weight
            loss += mel_loss
        if net_output["prosody_out"]["f0"] is not None:
            pred, targ = net_output["prosody_out"]["f0"], \
                        sample["f0_target"]
            
            mask = sample["f0_mask"]
            # Crop f0 to valid length
            crop_len = min(targ_mask.sum(1).max().long().item(), pred.size(1), targ.size(1))
            pred = pred[:,:crop_len].contiguous()
            targ = targ[:,:crop_len].contiguous()
            mask = mask[:,:crop_len].contiguous()
            
            
            if self.sentence_avg:
                f0_loss = ((self.criterion_l1(pred, targ).mean(-1) * mask).sum(1) / mask.sum(1)).sum()
            else:
                f0_loss = (self.criterion_l1(pred, targ).mean(-1) * mask).sum()
            #f0_loss = self.criterion_l1_f0(pred, targ) * mask
            f0_loss = f0_loss * self.f0_weight
            loss +=  f0_loss 


        if net_output["prosody_out"]["energy"] is not None:
            pred, targ = net_output["prosody_out"]["energy"], \
                               sample["energy_target"]
            
            mask = sample["energy_mask"]
            # Crop energy to valid length
            crop_len = min(mask.sum(1).max().item(), pred.size(1), targ.size(1))
            pred = pred[:,:crop_len].contiguous()
            targ = targ[:,:crop_len].contiguous()
            mask = mask[:,:crop_len].contiguous()
            
            if self.sentence_avg:
                energy_loss = ((self.criterion_l1(pred, targ).mean(-1) * mask).sum(1) / mask.sum(1)).sum()
            else:
                energy_loss = (self.criterion_l1(pred, targ).mean(-1) * mask).sum()
            #energy_loss = self.criterion_l1_energy(pred, targ) * mask
            energy_loss = energy_loss * self.energy_weight
            loss += energy_loss

        # if torch.isnan(loss):
        #     logger.error("NaN detected in final loss")
        #     logger.error(f"mel_loss: {mel_loss}, energy_loss: {energy_loss},f0_loss: {f0_loss},sc_loss: {sc_loss}, , nll_loss: {nll_loss}")
        #     # You could raise an exception here if desired
        #     raise ValueError(f"{loss}NaN detected in loss computation")
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        
        #print(f" loss: {loss}, mel_loss: {mel_loss}, f0_loss: {f0_loss}, energy_loss: {energy_loss}, hubert_loss: {hubert_loss}, sample_size:{sample_size}")
        
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "mel_loss": utils.item(mel_loss.data) if net_output["encoder_out_mel"] is not None else None,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "f0_loss": utils.item(f0_loss.data) if net_output["prosody_out"]["f0"] is not None else None,
            "energy_loss": utils.item(energy_loss.data) if net_output["prosody_out"]["energy"] is not None else None,
            "hubert_loss": utils.item(hubert_loss.data),
            
            }
        # if self.report_accuracy:
        #     n_correct, total, _, _, _ = self.compute_accuracy(model, net_output, sample)
        #     logging_output["n_correct"] = utils.item(n_correct.data)
        #     logging_output["total"] = utils.item(total.data)

        loss = loss / sample_size
        #assert 1==0, f"{loss.data}, {energy_loss.data}, {f0_loss.data}"
        return loss, sample_size, logging_output


    def compute_accuracy(self, model, net_output, sample, type_loss="hubert"):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, type_loss)
        mask = target.ne(self.padding_idx if type_loss == "hubert" else self.padding_idx_f0)
        
        # Get predictions
        predictions = lprobs.argmax(1)
        masked_preds = predictions.masked_select(mask)
        masked_targets = target.masked_select(mask)
        
        # Calculate accuracy
        n_correct = torch.sum(masked_preds.eq(masked_targets))
        total = torch.sum(mask)
        
        # Calculate true positives, false positives, and false negatives for each class
        n_classes = lprobs.size(-1)
        tp = torch.zeros(n_classes, device=lprobs.device)
        fp = torch.zeros(n_classes, device=lprobs.device)
        fn = torch.zeros(n_classes, device=lprobs.device)
        
        for c in range(n_classes):
            pred_c = masked_preds == c
            target_c = masked_targets == c
            tp[c] = (pred_c & target_c).sum()
            fp[c] = (pred_c & ~target_c).sum()
            fn[c] = (~pred_c & target_c).sum()
        
        return n_correct, total, tp, fp, fn
    def visualize_accuracy(self, model, net_output, sample, type_loss="hubert"):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, type_loss=type_loss)
        target = model.get_targets(sample, net_output, type_loss=type_loss)
    
        # Truncate lprobs and target to the same sequence length
        seq_len = min(lprobs.size(1), target.size(1))
        lprobs = lprobs[:, :seq_len]
        target = target[:, :seq_len].to(dtype=torch.int64)
        
        # Ignore prefix size if specified
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :, :].contiguous()
        mask = target.ne(self.padding_idx if type_loss == "hubert" else self.padding_idx_f0)
    
        # Get predictions and masked target
        predictions = lprobs.argmax(-1)  # Shape: [batch_size, seq_len]
        
        print("\nSample-wise Accuracy and Predictions:")
        for i in range(target.size(0)):  # Iterate over batch size
            sample_mask = mask[i]
            if sample_mask.any():  # Avoid empty masks
                correct = predictions[i][sample_mask].eq(target[i][sample_mask]).sum().item()
                total = sample_mask.sum().item()
                accuracy = correct / total
                print(f"Sample {i}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Predictions: {predictions[i].tolist()}")
                print(f"  Target: {target[i].tolist()}")
            else:
                print(f"Sample {i}: No valid tokens (mask is empty).")
        #assert 1==0, f"type loss: {type_loss}: \n "
    def compute_loss(self, model, net_output, sample, type_loss="hubert", reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, type_loss)
        
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx if type_loss == "hubert" else self.padding_idx_f0,
            reduce=reduce,
        )
        return loss, nll_loss
    def get_lprobs_and_target(self, model, net_output, sample, type_loss="hubert"):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, type_loss=type_loss)
        
        target = model.get_targets(sample, net_output, type_loss=type_loss)
       
        lprobs = lprobs[:, :min(lprobs.size(1), target.size(1))]
        target = target[:, :min(lprobs.size(1), target.size(1))]
        target = target.to(dtype=torch.int64)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        
        

        return lprobs.reshape(-1, lprobs.size(-1)), target.reshape(-1)

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        metrics.log_scalar("ntokens", ntokens, round=3)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("sample_size", sample_size, round=3)



        if logging_outputs[0]["mel_loss"] is not None:
            mel_loss = sum(log.get("mel_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "mel_loss", mel_loss , sample_size, round=5
            )
            metrics.log_scalar(
                "mel_loss_by_sample_size", mel_loss / sample_size , sample_size, round=5
            )
        if logging_outputs[0]["f0_loss"] is not None:
            f0_loss_sum = sum(log.get("f0_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "f0_loss", f0_loss_sum , sample_size, round=5
            )
            metrics.log_scalar(
                "f0_loss_by_sample_size", f0_loss_sum / sample_size, sample_size, round=5
            )
        if logging_outputs[0]["energy_loss"] is not None:
            energy_loss_sum = sum(log.get("energy_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "energy_loss", energy_loss_sum , sample_size, round=5
            )
            metrics.log_scalar(
                "energy_loss_by_sample_size", energy_loss_sum / sample_size, sample_size, round=5
            )

        if logging_outputs[0]["hubert_loss"] is not None:
            hubert_loss_sum = sum(log.get("hubert_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "hubert_loss", hubert_loss_sum , sample_size, round=5
            )

            metrics.log_scalar(
                "hubert_loss_by_sample_size", hubert_loss_sum /sample_size, sample_size, round=5
            )

        if "f0_n_correct" in logging_outputs[0]:
            n_correct = sum(log.get("f0_n_correct", 0) for log in logging_outputs)
            total = sum(log.get("f0_total", 0) for log in logging_outputs)
            metrics.log_scalar("f0_n_correct", n_correct)
            metrics.log_scalar("f0_total", total)
            metrics.log_derived(
                "f0_accuracy",
                lambda meters: round(
                    float(meters["f0_n_correct"].sum * 100.0 / meters["f0_total"].sum), 3
                )
                if meters["f0_total"].sum > 0
                else float("nan"),
            )

            # Add energy accuracy logging
        if "energy_n_correct" in logging_outputs[0]:
            n_correct = sum(log.get("energy_n_correct", 0) for log in logging_outputs)
            total = sum(log.get("energy_total", 0) for log in logging_outputs)
            metrics.log_scalar("energy_n_correct", n_correct)
            metrics.log_scalar("energy_total", total)
            metrics.log_derived(
                "energy_accuracy",
                lambda meters: round(
                    float(meters["energy_n_correct"].sum * 100.0 / meters["energy_total"].sum), 3
                )
                if meters["energy_total"].sum > 0
                else float("nan"),
            )

        # F1 score calculation for F0
        if "f0_tp" in logging_outputs[0]:
            tp = sum(log.get("f0_tp", 0) for log in logging_outputs)
            fp = sum(log.get("f0_fp", 0) for log in logging_outputs)
            fn = sum(log.get("f0_fn", 0) for log in logging_outputs)
            
            metrics.log_scalar("f0_tp", tp)
            metrics.log_scalar("f0_fp", fp)
            metrics.log_scalar("f0_fn", fn)
            
            metrics.log_derived(
                "f0_precision",
                lambda meters: round(
                    float(meters["f0_tp"].sum / (meters["f0_tp"].sum + meters["f0_fp"].sum + 1e-13)) * 100, 3
                )
            )
            
            metrics.log_derived(
                "f0_recall",
                lambda meters: round(
                    float(meters["f0_tp"].sum / (meters["f0_tp"].sum + meters["f0_fn"].sum + 1e-13)) * 100, 3
                )
            )
            
            metrics.log_derived(
                "f0_f1",
                lambda meters: round(
                    float(2 * meters["f0_tp"].sum / (2 * meters["f0_tp"].sum + meters["f0_fp"].sum + meters["f0_fn"].sum + 1e-13)) * 100, 3
                )
            )

        # F1 score calculation for Energy
        if "energy_tp" in logging_outputs[0]:
            tp = sum(log.get("energy_tp", 0) for log in logging_outputs)
            fp = sum(log.get("energy_fp", 0) for log in logging_outputs)
            fn = sum(log.get("energy_fn", 0) for log in logging_outputs)
            
            metrics.log_scalar("energy_tp", tp)
            metrics.log_scalar("energy_fp", fp)
            metrics.log_scalar("energy_fn", fn)
            
            metrics.log_derived(
                "energy_precision",
                lambda meters: round(
                    float(meters["energy_tp"].sum / (meters["energy_tp"].sum + meters["energy_fp"].sum + 1e-13)) * 100, 3
                )
            )
            
            metrics.log_derived(
                "energy_recall",
                lambda meters: round(
                    float(meters["energy_tp"].sum / (meters["energy_tp"].sum + meters["energy_fn"].sum + 1e-13)) * 100, 3
                )
            )
            
            metrics.log_derived(
                "energy_f1",
                lambda meters: round(
                    float(2 * meters["energy_tp"].sum / (2 * meters["energy_tp"].sum + meters["energy_fp"].sum + meters["energy_fn"].sum + 1e-13)) * 100, 3
                )
            )


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag_list, y_mag_list, sentence_avg):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        loss = 0.
        for x_mag, y_mag in zip(x_mag_list, y_mag_list):
            loss_one_sample = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            loss += loss_one_sample if sentence_avg else loss_one_sample * len(y_mag)
        return loss