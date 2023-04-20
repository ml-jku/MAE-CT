import einops
import torch

from schedules import schedule_from_kwargs


class MaskGenerator:
    def __init__(
            self,
            mask_ratio=None,
            mask_ratio_schedule=None,
            seed=None,
            single_mask_seed=None,
            update_counter=None,
    ):
        assert (mask_ratio is not None) ^ (mask_ratio_schedule is not None)
        self.constant_mask_ratio = mask_ratio
        self.update_counter = update_counter
        if mask_ratio_schedule is not None:
            assert update_counter is not None
            self.mask_ratio_schedule = schedule_from_kwargs(mask_ratio_schedule, update_counter=self.update_counter)
        else:
            self.mask_ratio_schedule = None

        self.seed = seed
        self.generators = {}
        self.single_mask_seed = single_mask_seed
        self._single_mask_noise = None

    @property
    def mask_ratio(self):
        if self.constant_mask_ratio is not None:
            return self.constant_mask_ratio
        return self.mask_ratio_schedule.get_value(self.update_counter.cur_checkpoint)

    def get_single_mask_noise(self, x):
        assert self.single_mask_seed is not None
        if self._single_mask_noise is None:
            if self.single_mask_seed is not None:
                generator = torch.Generator(device=x.device).manual_seed(self.single_mask_seed)
            else:
                generator = None
            self._single_mask_noise = self.generate_noise(x, generator=generator)
        return self._single_mask_noise

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}({self._base_param_str})"

    @property
    def _base_param_str(self):
        base_str = f"mask_ratio={self.mask_ratio}"
        if self.seed is not None:
            base_str += f",seed={self.seed}"
        if self.single_mask_seed is not None:
            base_str += f",single_mask_seed={self.single_mask_seed}"
        return base_str

    def get_generator(self, x):
        if self.seed is None:
            return None
        device_str = str(x.device)
        # generator doesn't support moving between devices
        # (and also copying state is not possible as gpu and cpu generators have different state sizes)
        if device_str not in self.generators:
            generator = torch.Generator(device=x.device).manual_seed(self.seed)
            self.generators[device_str] = generator
            return generator
        return self.generators[device_str]

    def generate_noise(self, x, generator=None):
        raise NotImplementedError

    @staticmethod
    def sort_noise(noise):
        noise = einops.rearrange(noise, "N H W -> N (H W)")
        ids_restore = torch.argsort(noise, dim=1)
        return ids_restore

    def get_mask(self, x, single_mask=False):
        N, D, H, W = x.shape
        L = H * W
        len_keep = int(L * (1 - self.mask_ratio))

        # generate noise
        if single_mask:
            noise = self.get_single_mask_noise(x).to(x.device)
        else:
            noise = self.generate_noise(x, generator=self.get_generator(x))

        # reshape x from "image" to sequence
        x = einops.rearrange(x, "N D H W -> N (H W) D")

        # sort noise for each sample
        ids_shuffle = self.sort_noise(noise)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
