# ema.py
import contextlib
import torch

class EMA:
    """
    Exponential Moving Average of model parameters.
    - Tracks only trainable parameters (requires_grad=True).
    - Keeps fp32 shadow weights.
    - Works with DDP (pass model.module to ctor if wrapped).
    - Provides a safe context manager to temporarily apply EMA weights.
    """
    def __init__(self, model, decay=0.9999, device=None, use_num_updates=True, only_trainable=True):
        """
        Args:
            model: torch.nn.Module (DDP: pass model.module)
            decay: base decay (e.g., 0.999–0.9999)
            device: where to keep shadow weights; None keeps them on param.device.
                    Use "cpu" to save VRAM (slightly slower updates due to copies).
            use_num_updates: if True, warmup the decay by number of updates.
            only_trainable: if True, track only params with requires_grad=True.
        """
        self.decay = float(decay)
        self.device = torch.device(device) if device is not None else None
        self.use_num_updates = bool(use_num_updates)
        self.only_trainable = bool(only_trainable)
        self.num_updates = 0

        # name -> tensor (fp32)
        self.shadow = {}
        self.backup = None

        for name, p in model.named_parameters():
            if (not p.requires_grad) and self.only_trainable:
                continue
            if p.data.dtype == torch.float16 or p.data.dtype == torch.bfloat16:
                data = p.data.float()
            else:
                data = p.data.detach().clone()
            if self.device is not None:
                data = data.to(self.device)
            self.shadow[name] = data

    def _get_decay(self):
        if not self.use_num_updates:
            return self.decay
        # timm-style warmup toward target decay
        self.num_updates += 1
        warmup_decay = 1.0 - 1.0 / (self.num_updates + 1.0)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def update(self, model):
        """Call after each optimizer.step()."""
        d = self._get_decay()
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            if not p.requires_grad and self.only_trainable:
                continue
            # bring param to shadow device/dtype
            cur = p.data
            if cur.dtype != torch.float32:
                cur = cur.float()
            if self.device is not None and cur.device != self.device:
                cur = cur.to(self.device)
            # ema = d * ema + (1 - d) * param
            self.shadow[name].mul_(d).add_(cur, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model):
        """Overwrite model params with EMA weights (in-place)."""
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            ema_w = self.shadow[name]
            if p.data.dtype != torch.float32:
                tgt = ema_w.to(device=p.data.device, dtype=torch.float32)
                p.data.copy_(tgt).to(dtype=p.data.dtype)
            else:
                p.data.copy_(ema_w.to(device=p.data.device, dtype=p.data.dtype))

    @torch.no_grad()
    def store(self, model):
        """Save current (non-EMA) weights so we can restore later."""
        self.backup = {name: p.data.detach().clone()
                       for name, p in model.named_parameters()
                       if name in self.shadow}

    @torch.no_grad()
    def restore(self, model):
        """Restore the weights saved by .store()."""
        if self.backup is None:
            return
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = None

    @contextlib.contextmanager
    def apply_to(self, model):
        """
        Context manager:
        with ema.apply_to(model):
            ... # run eval/sampling with EMA weights
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    # ----- checkpoint I/O -----
    def state_dict(self):
        # always move to CPU for portability
        cpu_shadow = {k: v.detach().to("cpu") for k, v in self.shadow.items()}
        return {
            "decay": self.decay,
            "use_num_updates": self.use_num_updates,
            "only_trainable": self.only_trainable,
            "num_updates": self.num_updates,
            "shadow": cpu_shadow,
        }

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.use_num_updates = bool(state.get("use_num_updates", True))
        self.only_trainable = bool(state.get("only_trainable", True))
        self.num_updates = int(state.get("num_updates", 0))
        loaded_shadow = state["shadow"]
        # keep existing devices/dtypes for efficiency
        for k, v in loaded_shadow.items():
            if k in self.shadow:
                dev = self.shadow[k].device
                self.shadow[k] = v.to(dev).float()