"""
Filename: 3dgs.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of 3DGS based on the work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 
This implementation is modified from the nerfstudio GaussianSplattingModel.

- Original work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
- Codebase reference: nerfstudio GaussianSplattingModel (https://github.com/nerfstudio-project/nerfstudio/blob/gaussian-splatting/nerfstudio/models/gaussian_splatting.py)

Original paper: https://arxiv.org/abs/2308.04079
"""

from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

from models.gaussians.basics import *

logger = logging.getLogger()


class VanillaGaussians(nn.Module):

    def __init__(
            self,
            class_name: str,
            ctrl: OmegaConf,
            reg: OmegaConf = None,
            networks: OmegaConf = None,
            scene_scale: float = 30.,
            scene_origin: torch.Tensor = torch.zeros(3),
            num_train_images: int = 300,
            device: torch.device = torch.device("cuda"),
            **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0

        self.device = device
        self.ball_gaussians = self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)

        # for evaluation
        self.in_test_set = False

        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)

        # 降雨参数
        self.diameter = torch.zeros(1, 1, device=self.device)
        self.velocity = torch.zeros(1, 1, device=self.device)
        self.direction = torch.zeros(1, 3, device=self.device)
        self.exposure_time = 0.05
        self.delta_t = 0.1

    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        self._means = Parameter(init_means)

        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors)  # float range [0, 1]
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)

    @property
    def shs_0(self):
        return self._features_dc

    @property
    def shs_rest(self):
        return self._features_rest

    @property
    def num_points(self):
        return self._means.shape[0]

    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)

    @property
    def get_quats(self):
        return self.quat_act(self._quats)

    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def preprocess_per_train_step(self, step: int):
        self.step = step

    def postprocess_per_train_step(
            self,
            step: int,
            optimizer: torch.optim.Optimizer,
            radii: torch.Tensor,
            xys_grad: torch.Tensor,
            last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
            self,
            radii: torch.Tensor,
            xys_grad: torch.Tensor,
            last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask

            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix + "xyz": [self._means],
            self.class_prefix + "sh_dc": [self._features_dc],
            self.class_prefix + "sh_rest": [self._features_rest],
            self.class_prefix + "opacity": [self._opacities],
            self.class_prefix + "scaling": [self._scales],
            self.class_prefix + "rotation": [self._quats],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                    self.step < self.ctrl_cfg.stop_split_at
                    and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None

                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()

                splits = (
                        self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                        self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                ) = self.dup_gaussians(dups)

                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(
                    torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(
                    torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(
                    torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix + "opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                    torch.exp(self._scales).max(dim=-1).values >
                    self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
                self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats

    def get_gaussians(self, cam: dataclass_camera, class_name="Not Rain") -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask

        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        activated_means = self._means
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        activated_colors = rgbs

        if class_name == 'Rain':
            # 计算雨纹长度
            length = self.get_length()
            # 计算需要雨滴需要复制的次数，取整
            num_duplication = torch.ceil(length / self.diameter).to(torch.int32)
            # 取num_duplication中的最大值
            max_duplication = torch.max(num_duplication).item()
            n = max_duplication - 1
            # 生成掩码，过滤掉原本复制次数小于max_duplication的雨滴
            mask = (torch.arange(n, device=self.device)[:, None] < num_duplication).flatten()
            # 计算偏移量，每次平移diameter/2, 直到所有雨滴都被复制完毕
            # 生成形状为 (n, 1) 的偏移倍数
            offset_multipliers = torch.arange(1, n+1, device=self.device).unsqueeze(
                1)  # [n, 1]
            # 形状为 (1, N) 的 diameter
            diameters = self.diameter.unsqueeze(0)  # [1, N]

            # 利用广播机制计算偏移量
            offsets = offset_multipliers * diameters  # [n, N]
            offsets = offsets.unsqueeze(-1)  # [n, N, 1]
            # 沿着self.direction方向扩展雨滴的均值
            direction = self.direction.unsqueeze(0)  # [1, 1, 3]
            expanded_means = activated_means.unsqueeze(0)  # [1, N, 3]
            expanded_means = expanded_means.repeat(n, 1, 1)  # [n, N, 3]
            offsets = offsets * direction
            expanded_means += offsets  # 广播偏移量
            # 合并所有副本
            activated_means = expanded_means.view(-1, 3)  # [n*N, 3]
            activated_means = activated_means[mask]

            # 其他属性
            activated_opacities = activated_opacities.repeat(n, 1)
            activated_opacities = activated_opacities[mask]
            activated_scales = activated_scales.repeat(n, 1)
            activated_scales = activated_scales[mask]
            activated_rotations = activated_rotations.repeat(n, 1)
            activated_rotations = activated_rotations[mask]
            activated_colors = activated_colors.repeat(n, 1)
            activated_colors = activated_colors[mask]
            self.update_rain()

        # collect gaussians information
        gs_dict = dict(
            _means=activated_means,
            _opacities=activated_opacities,
            _rgbs=activated_colors,
            _scales=activated_scales,
            _quats=activated_rotations,
        )

        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")

        return gs_dict

    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                                          torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w

        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1 - 1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1 - opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        return loss_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg

    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors

        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }

    def add_rain(self, init_pose) -> None:
        """
        add rain to the model
        """
        gaussian_dict = {}
        mean_rgb = np.array([200, 200, 200]) / 255
        # 降雨区域, waymo coordinate system: x front, y left, z up
        x_min, x_max = 0, 40
        y_min, y_max = -10, 10
        z_min, z_max = -3, 17
        # 雨滴数量, 雨滴直径和速度
        density = 1
        num_points = (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * density
        diameter = torch.distributions.Gamma(6.648, 1 / 0.166).sample((num_points,)).to(self.device)
        self.velocity = 3.197 * torch.pow(diameter, 0.672)
        self.diameter = diameter / 1000
        self.get_direction()

        # 生成平移量,直接生成三维均匀分布采样点（每行是一个三维点）
        low = torch.tensor([x_min, y_min, z_min], device=self.device)
        high = torch.tensor([x_max, y_max, z_max], device=self.device)
        # 生成 [0,1) 均匀分布后缩放到 [low, high)
        translation = (high - low) * torch.rand((num_points, 3), device=self.device) + low
        gaussian_dict["means"] = translation + init_pose[:3, 3]

        # 生成缩放，创建一个和translation一样长度，和self.scales一样宽度的全1tensor
        gaussian_dict["scales"] = torch.ones((num_points, 3), dtype=torch.float32, device=self.device)
        gaussian_dict["scales"][:, :2] = torch.tensor(self.diameter, dtype=torch.float32, device=self.device).view(-1,
                                                                                                                   1).repeat(
            1, 2)
        gaussian_dict["scales"][:, 2] = torch.where(
            self.diameter < 1,
            self.diameter,
            self.diameter * (1.07 - 0.07 * self.diameter)
        )
        gaussian_dict["scales"] = torch.log(gaussian_dict["scales"])

        # 生成旋转四元数
        gaussian_dict["quats"] = torch.zeros((num_points, 4), dtype=torch.float32, device=self.device)
        gaussian_dict["quats"][:, 0] = 1

        # 球谐函数
        mean_rgb_tensor = torch.tensor(mean_rgb, dtype=torch.float32, device=self.device)
        gaussian_dict["features_dc"] = torch.ones((num_points, 3), dtype=torch.float32,
                                                  device=self.device) * mean_rgb_tensor
        gaussian_dict["features_rest"] = torch.zeros((num_points, 15, 3), dtype=torch.float32, device=self.device)

        # 透明度
        opacities = self.diameter / (self.velocity * self.exposure_time)
        # 使用logit函数将opacities转换为[-1, 1]范围内的值，因为后面会用sigmoid函数处理
        opacities = torch.logit(opacities)
        gaussian_dict["opacities"] = torch.tensor(opacities, dtype=torch.float32, device=self.device).view(-1, 1)

        self._means = Parameter(gaussian_dict["means"])
        self._scales = Parameter(gaussian_dict["scales"])
        self._quats = Parameter(gaussian_dict["quats"])
        self._features_dc = Parameter(gaussian_dict["features_dc"])
        self._features_rest = Parameter(gaussian_dict["features_rest"])
        self._opacities = Parameter(gaussian_dict["opacities"])

    def update_rain(self) -> None:
        # 计算位移
        translation = self.direction * self.velocity.unsqueeze(1) * self.delta_t
        # 更新位置
        self._means += translation

    def get_length(self) -> torch.Tensor:
        # 单位为 m
        return self.velocity * self.exposure_time

    def get_direction(self) -> None:
        # 提取四元数分量
        w, x, y, z = self._quats[:, 0], self._quats[:, 1], self._quats[:, 2], self._quats[:, 3]
        # 计算旋转矩阵
        dir_z_x = 2 * (x * z + w * y)
        dir_z_y = 2 * (y * z - w * x)
        dir_z_z = -1 + 2 * (x * x + y * y)
        self.direction = torch.stack([dir_z_x, dir_z_y, dir_z_z], dim=1)
        self.direction / self.direction.norm(dim=-1, keepdim=True)
