from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Mapping, final

import lightning as pl2
import torch
import torchinfo
from torch import isnan, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from ..module.lambda_layer import LambdaLayer
from ..module.model_template import ModelInputArgs


class MInterface(pl2.LightningModule, ABC):
    required_output_keys: list[str]
    get_stats_fn: (
        Callable[
            [MInterface, torch.Tensor, torch.Tensor, dict],
            dict[str, torch.Tensor],
        ]
        | None
    )

    def __init__(
        self,
        /,
        *,
        model_class: type[torch.nn.Module] | Callable[..., torch.nn.Module],
        model_args: dict[str, Any],
        model_common_args: ModelInputArgs,
        loss: torch.nn.modules.loss._Loss | Sequence[torch.nn.modules.loss._Loss],
        loss_weights: Sequence[float] | None = None,
        multiloss_weights: Sequence[float] | None = None,
        optimizer_class: type[Optimizer] = AdamW,
        optimizer_args: dict[str, Any] | None = None,
        lr_scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_args: dict[str, Any] | None = None,
        ckpt_path: str = "",
        log_grad: bool = False,
        log_norm: bool = False,
        summary_verbose: bool = True,
        get_stats_fn: Callable | None = None,
    ):
        super().__init__()

        # Check and configure loss
        if isinstance(loss, Sequence):
            assert multiloss_weights is None or (
                isinstance(multiloss_weights, Sequence)
                and len(loss) == len(multiloss_weights)
            ), f"When specifying multiple losses, you must also specify the corresponding loss weights to be a sequence or None, but got {loss} and {multiloss_weights}"
        elif isinstance(loss, torch.nn.modules.loss._Loss):
            assert (
                multiloss_weights is None
            ), f"When specifying a single loss, you should not specify the loss weights, but got {loss} and {multiloss_weights}"
        self.loss = loss
        self.loss_weights = (
            nn.Parameter(torch.tensor(loss_weights, device=self.device))
            if loss_weights is not None
            else None
        )
        self.multiloss_weights = multiloss_weights
        self.configure_loss()

        self.optmizer_class = optimizer_class
        self.optimizer_args = optimizer_args
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_args = lr_scheduler_args

        # Instantiate main model and possibly load checkpoint
        if model_common_args.num_channels is None:
            from ...utils.channel_enum import NUM_ELECTRODES as num_channels

            model_common_args.num_channels = num_channels
        self.model = model_class(**model_args, **model_common_args.model_dump())

        # Instantiate pre_model and post_model
        self.stage = "train"
        self.pre_model: torch.nn.Module = LambdaLayer(lambda x: x["eeg"])
        self.post_model: torch.nn.Module = torch.nn.Identity()

        # Configure the input/output of the main model.
        self._required_inputs = self.configure_input()
        self.get_input_size(**model_common_args.model_dump())
        self._output_keys = self.configure_output()

        self.summary_verbose = int(summary_verbose or False)
        summary: torchinfo.ModelStatistics = torchinfo.summary(
            self.model,
            input_size=list(self.input_size.values()),
            verbose=self.summary_verbose,
        )
        self.output_size = summary.summary_list[0].output_size

        self.ckpt_path = ckpt_path  # Store checkpoint path for later use
        self.log_grad = log_grad
        self.log_norm = log_norm

        self.get_stats_fn = get_stats_fn

    @final
    def get_input_size(self, /, **kwargs) -> Mapping[str, tuple[int]]:
        """
        Get the input size for each required input type based on the model's forward method.

        Args:
            kwargs: Additional arguments to determine input dimensions.

        Returns:
            list[tuple[int | None, ...]]: A list of input sizes in the order specified by self._required_inputs.
        """
        # Explicitly ensure dictionary insert order
        input_sizes = OrderedDict()

        # Determine EEG/EXG input size
        input_length = kwargs["fs"] * kwargs["window_length"]

        num_channels = kwargs["num_channels"]

        for required_input in self._required_inputs:
            # Add EEG/EXG input size if required
            if required_input == "eeg":
                input_sizes["eeg"] = (1, input_length, num_channels)

            # Add audio input size if required
            if "audio" == required_input:
                input_sizes["audio"] = (
                    1,
                    input_length,
                    kwargs["num_audio_features"],
                    1,
                )

            # Add label input size if required
            if "label" == required_input:
                input_sizes["label"] = (1,)

        self.input_size = input_sizes

        return input_sizes

    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.

        Args:
            data (dict): A dictionary containing the input data. Keys may include 'eeg', 'audio', and 'label'.

        Returns:
            tuple: The output of the model after processing through pre_model, model, and post_model,
                along with any additional required outputs from data.
        """
        # Prepare inputs based on the required order from configure_input
        inputs = []
        for input_type in self._required_inputs:
            if input_type == "eeg":
                # Process EEG/EXG input through pre_model
                inputs.append(self.pre_model(data))
            elif input_type == "audio":
                assert (
                    "audio" in data
                ), f"EEG_CLASSIFY_BASE_DATASET:FORWARD:ASSERTION:VALUE_ERROR: audio input is required, but not found in data. Available keys: {data.keys()}"
                # Extract audio input directly from data
                inputs.append(data["audio"])
            elif input_type == "label":
                assert (
                    "label" in data
                ), f"EEG_CLASSIFY_BASE_DATASET:FORWARD:ASSERTION:VALUE_ERROR: label input is required, but not found in data. Available keys: {data.keys()}"
                # Extract label input directly from data
                inputs.append(data["label"])

        # Pass the inputs to the model in the required order
        model_outputs = self.model(*inputs)

        # Ensure model_outputs is a tuple
        if not isinstance(model_outputs, tuple):
            model_outputs = (model_outputs,)

        # Process the first output (eeg_hat) through post_model
        eeg_output = self.post_model(model_outputs[0])

        # Collect all outputs
        outputs = [eeg_output]

        # Add additional outputs from the model if they exist
        if len(self._output_keys) > 1:
            outputs.extend(model_outputs[1:])

        # Add required outputs from data if not already in model outputs
        for key in self.required_output_keys:
            if key not in self._output_keys:
                outputs.append(data[key])

        return tuple(outputs)

    @abstractmethod
    def get_stats(
        self, outputs: torch.Tensor, targets: torch.Tensor, /, *, meta: dict
    ) -> dict[str, torch.Tensor]:
        """get_stats. This method will be called during `training_step`, `validation_step` and `test_step`. It should calculate the statistics of the model's output and the target.

        Args:
            output (torch.Tensor): the output of the model
            target (torch.Tensor): the target/label

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the calculated statistics.

        Example:
            ```python
            def get_stats(self, output: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
                self.log('accuracy', (output == target).float().mean())
                return {'accuracy': (output == target).float().mean()}
            ```
        """
        pass

    def training_closure(self, *args):
        """If you want to do any arbitary modification to the data before or after the calling of `self.forward`, override this method. This method will be called during `training_step`, and therefore, also `validation_step` and `test_step` in our logics. The default implementation is to call `self.forward` directly."""

        # Call the forward method of the model with the provided arguments
        return self.forward(*args)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = torch.zeros(1, device=self.device)
        batch_size = 0
        for data in batch.values():
            outputs = self.training_closure(data)
            loss += self.loss_fn(*outputs)
            batch_size += outputs[0].shape[0]
            self.get_stats(
                *outputs,
                meta=data["meta"],
            )
            if self.get_stats_fn:
                self.get_stats_fn(self, *outputs, meta=data["meta"])  # type: ignore

        loss /= batch_size
        self.log(
            f"{self.stage}/loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            enable_graph=False,
        )

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def test_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def on_train_epoch_start(self) -> None:
        self.stage = "train"
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.stage = "val"
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.stage = "test"
        return super().on_test_epoch_start()

    @final
    def configure_loss(self):
        """Configure the loss function. If multiple losses are provided, the loss function will be a sum of all the losses. If a single loss is provided, the loss function will be the loss itself. The loss function will be stored in `self.loss_fn`. When calling `self.loss_fn`, always put the output of the model first and the target/label as the second.

        Returns:
            None, the configured loss will be accessible through `self.loss_fn`
        """
        if isinstance(self.loss, Sequence):
            # add a closure variable to let static type checker know the type of loss_fn
            if self.multiloss_weights is None:
                self.multiloss_weights = [1] * len(self.loss)

            self.multiloss_weights = torch.tensor(
                self.multiloss_weights, device=self.device
            )

            if self.loss_weights is not None:

                def loss_fn(  # type: ignore
                    *args: torch.Tensor
                    | int
                    | str
                    | Sequence[torch.Tensor | int | str],
                ):
                    loss = torch.zeros(1, device=self.device)
                    for loss_fn, weight in zip(self.loss, self.multiloss_weights):  # type: ignore
                        loss += loss_fn(*args, self.loss_weights).sum() * weight
                    assert not isnan(loss), (
                        "MODEL_INTERFACE:LOSS_FN:ASSERTION:VALUE_ERROR: Loss function returned NaN. "
                        "This usually indicates a problem with the model or the data. "
                        "Please check your model and data for any issues."
                    )
                    return loss

            else:

                def loss_fn(  # type: ignore
                    *args: torch.Tensor
                    | int
                    | str
                    | Sequence[torch.Tensor | int | str],
                ):
                    loss = torch.zeros(1, device=self.device)
                    for loss_fn, weight in zip(self.loss, self.multiloss_weights):  # type: ignore
                        loss += loss_fn(*args).sum() * weight
                    assert not isnan(loss), (
                        "MODEL_INTERFACE:LOSS_FN:ASSERTION:VALUE_ERROR: Loss function returned NaN. "
                        "This usually indicates a problem with the model or the data. "
                        "Please check your model and data for any issues."
                    )
                    return loss

        else:
            if self.loss_weights is not None:

                def loss_fn(*args: torch.Tensor | int | str):
                    loss = self.loss(*args, self.loss_weights).sum()  # type: ignore
                    assert not isnan(loss), (
                        "MODEL_INTERFACE:LOSS_FN:ASSERTION:VALUE_ERROR: Loss function returned NaN. "
                        "This usually indicates a problem with the model or the data. "
                        "Please check your model and data for any issues."
                    )
                    return loss

            else:

                def loss_fn(*args: torch.Tensor | int | str):
                    loss = self.loss(*args).sum()  # type: ignore
                    assert not isnan(loss), (
                        "MODEL_INTERFACE:LOSS_FN:ASSERTION:VALUE_ERROR: Loss function returned NaN. "
                        "This usually indicates a problem with the model or the data. "
                        "Please check your model and data for any issues."
                    )
                    return loss

            loss_fn.__repr__ = lambda: f"{self.loss.__repr__().split('(')[0]}"

        self.loss_fn = loss_fn

    def on_after_backward(self):
        if getattr(self, "log_grad", False):
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.log(
                        f"grad_norm2/{name}",
                        param.grad.detach().data.norm(2).item(),
                        on_epoch=False,
                        batch_size=1,
                        enable_graph=False,
                    )
                    self.log(
                        f"param_norm2/{name}",
                        param.detach().data.norm(2).item(),
                        on_epoch=False,
                        batch_size=1,
                        enable_graph=False,
                    )
        super().on_after_backward()

    def on_before_backward(self, loss):
        if getattr(self, "log_norm", False):
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.log(
                        f"param_norm2/{name}",
                        param.detach().data.norm(2).item(),
                        on_epoch=False,
                        batch_size=1,
                        enable_graph=False,
                    )
        return super().on_before_backward(loss)

    @final
    def configure_input(self) -> list[str]:
        """
        Configure the input requirements for the model based on the forward method's input signature.

        Returns:
            list[str]: A sequence of strings indicating the required inputs in the order
                       specified by the model's forward method signature.
                       Possible values: 'eeg', 'audio', 'label'.
        """

        required_inputs = ["eeg"]

        # Inspect the forward method of the model
        if hasattr(self.model, "forward"):
            forward_signature = inspect.signature(self.model.forward)
            forward_params = forward_signature.parameters

            # Check for required inputs based on parameter names
            for param_name in forward_params:
                if param_name in ["env", "mel", "audio"]:
                    required_inputs.append("audio")
                elif param_name == "label":
                    required_inputs.append("label")

        return required_inputs

    @final
    def configure_output(self) -> list[str]:
        """
        Configure the output requirements for the model based on the forward method's output signature.

        Returns:
            list[str]: A sequence of strings indicating the outputs of the model.
                    Possible values: 'eeg_hat', 'audio_hat', 'label_hat'.
        """

        output_keys = ["eeg"]  # EEG output is always present by default

        # Inspect the forward method of the model
        if hasattr(self.model, "forward"):
            forward_signature = inspect.signature(self.model.forward)
            return_annotation = forward_signature.return_annotation

            # Check if the return annotation is a tuple or a single value
            if hasattr(return_annotation, "__args__") and isinstance(
                return_annotation.__args__, tuple
            ):
                # If the return type is a tuple, inspect its elements
                for output_type in return_annotation.__args__[1:]:
                    if "audio" in str(output_type).lower():
                        output_keys.append("audio")
                    elif "label" in str(output_type).lower():
                        output_keys.append("label")

        return output_keys

    def configure_optimizers(self):  # type: ignore

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "Norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        grouped_params = [
            {"params": decay, "weight_decay": self.weight_decay, "lr": self.lr},
            {
                "params": no_decay,
                "weight_decay": 0,
                "lr": self.lr,
            },
        ]

        optimizer = self.optmizer_class(grouped_params)  # type: ignore

        # return optimizer

        scheduler = {
            "scheduler": self.lr_scheduler_class(
                optimizer, **self.lr_scheduler_args if self.lr_scheduler_args else {}
            ),  # type: ignore
            "monitor": "val/loss",  # ⚠️ 这里必须指定你验证时 log 的指标名
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def lr(self) -> float:
        assert self.optimizer_args is not None, "optimizer_args is None"
        assert "lr" in self.optimizer_args, "'lr' key is missing in optimizer_args"

        return self.optimizer_args["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        assert isinstance(value, float), f"lr must be a float, but got {type(value)}"
        assert self.optimizer_args is not None, "optimizer_args is None"
        assert "lr" in self.optimizer_args, "'lr' key is missing in optimizer_args"
        self.optimizer_args["lr"] = value

    @property
    def weight_decay(self) -> float:
        assert self.optimizer_args is not None, "optimizer_args is None"
        assert (
            "weight_decay" in self.optimizer_args
        ), "'weight_decay' key is missing in optimizer_args"
        return self.optimizer_args["weight_decay"]

    @weight_decay.setter
    def weight_decay(self, value: float) -> None:
        assert isinstance(
            value, float
        ), f"weight_decay must be a float, but got {type(value)}"
        assert self.optimizer_args is not None, "optimizer_args is None"
        assert (
            "weight_decay" in self.optimizer_args
        ), "'weight_decay' key is missing in optimizer_args"
        self.optimizer_args["weight_decay"] = value
