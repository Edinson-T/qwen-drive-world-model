# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The perception model: BEV encoder + heads, fed from a Qwen3.5 VLM.

The perception weights are released separately from the VLM. Attach a loaded
``Qwen-Drive-1.0-4B`` VLM with :meth:`QwenDrivePerception.attach` and call
:meth:`QwenDrivePerception.infer` with processor outputs and ``img_metas``.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_perception import (
    DET_PC_RANGE,
    DET_VOXEL_SIZE,
    FRUSTUM_RANGE,
    FRUSTUM_SIZE,
    MAP_XBOUND,
    MAP_YBOUND,
    NUPLAN_OCC_PC_RANGE,
    NUPLAN_OCC_VOXEL_SIZE,
    NUSCENES_OCC_PC_RANGE,
    NUSCENES_OCC_VOXEL_SIZE,
    QwenDrivePerceptionConfig,
)
from .fpn import SimpleFPN
from .heads import BEVFormerHead
from .view_transform import DepthNet, Uni3DVoxelPoolDepth

__all__ = ["QwenDrivePerception", "BEVFormerModelV2"]


def _compress_feature_map(
    feature: torch.Tensor,
    *,
    bev_h: int,
    bev_w: int,
    output_hw: tuple[int, int],
    output_channels: int | None,
) -> torch.Tensor:
    """Convert a BEV token/map tensor into ``[B, C, H, W]`` float32."""
    if feature.ndim == 3:
        if feature.shape[1] == bev_h * bev_w:
            batch, length, channels = feature.shape
        elif feature.shape[0] == bev_h * bev_w:
            # The detection decoder returns [sequence, batch, channels].
            feature = feature.transpose(0, 1)
            batch, length, channels = feature.shape
        else:
            raise ValueError(
                f"cannot interpret BEV feature shape {tuple(feature.shape)} "
                f"for a {bev_h}x{bev_w} grid"
            )
        if length != bev_h * bev_w:
            raise ValueError(
                f"expected {bev_h * bev_w} BEV tokens, got {length} tokens "
                f"with shape {tuple(feature.shape)}"
            )
        feature = feature.view(batch, bev_h, bev_w, channels).permute(0, 3, 1, 2)
    elif feature.ndim != 4:
        raise ValueError(f"BEV feature must be 3D or 4D, got {tuple(feature.shape)}")

    feature = F.adaptive_avg_pool2d(feature.float(), output_hw)
    if output_channels is not None and feature.shape[1] != output_channels:
        in_channels = feature.shape[1]
        if in_channels % output_channels:
            raise ValueError(
                f"cannot reduce {in_channels} channels to {output_channels} without "
                "a learned projection"
            )
        feature = feature.view(
            feature.shape[0],
            output_channels,
            in_channels // output_channels,
            *feature.shape[-2:],
        ).mean(dim=2)
    return feature


class BEVFormerModelV2(nn.Module):
    """LLM-feature adaptor + UVTR view transform + perception heads."""

    def __init__(self, config: QwenDrivePerceptionConfig):
        super().__init__()
        self.config = config
        det_pc_range = config.det_pc_range
        det_voxel_size = list(config.det_voxel_size)
        occ_pillar_h = config.occ_pillar_h

        # Main stream: the LLM's image-token hidden states.
        self.adaptor = SimpleFPN(
            dim=config.llm_dim, out_channels=config.embed_dim, scale_factors=(4.0, 2.0, 1.0, 0.5)
        )

        # UVTR stream: ViT patch features -> depth -> voxel volume.
        self.vit_neck = SimpleFPN(dim=config.vit_dim, out_channels=config.embed_dim, scale_factors=(1.0,))
        uvtr_voxel_size = list(det_voxel_size)
        uvtr_voxel_size[2] = (det_pc_range[5] - det_pc_range[2]) / occ_pillar_h
        uvtr_voxel_shape = [
            int((det_pc_range[3] - det_pc_range[0]) / det_voxel_size[0]),
            int((det_pc_range[4] - det_pc_range[1]) / det_voxel_size[1]),
            occ_pillar_h,
        ]
        self.view_trans = Uni3DVoxelPoolDepth(
            pc_range=det_pc_range,
            voxel_size=uvtr_voxel_size,
            voxel_shape=uvtr_voxel_shape,
            frustum_range=config.frustum_range,
            frustum_size=config.frustum_size,
            embed_dim=config.embed_dim,
        )
        self.uvtr_query_proj = nn.Conv2d(config.embed_dim * occ_pillar_h, config.embed_dim, kernel_size=1)
        self.depth_net = DepthNet(
            config.embed_dim, config.embed_dim, self.view_trans.depth_dim, aspp_mid_channels=96
        )

        det_grid_conf = {
            "xbound": [det_pc_range[0], det_pc_range[3], det_voxel_size[0]],
            "ybound": [det_pc_range[1], det_pc_range[4], det_voxel_size[1]],
            "zbound": [det_pc_range[2], det_pc_range[5], det_voxel_size[2]],
        }
        map_grid_conf = {"xbound": list(config.map_xbound), "ybound": list(config.map_ybound), "zbound": [-10.0, 10.0, 20.0]}
        self.head = BEVFormerHead(
            bev_h=config.bev_h,
            bev_w=config.bev_w,
            num_query=config.num_query,
            num_classes=config.det_num_classes,
            embed_dims=config.embed_dim,
            code_size=config.code_size,
            det_pc_range=det_pc_range,
            det_voxel_size=det_voxel_size,
            map_grid_conf=map_grid_conf,
            det_grid_conf=det_grid_conf,
            occ_pc_range=config.nuscenes_occ_pc_range,
            nuplan_occ_pc_range=config.nuplan_occ_pc_range,
            occ_voxel_size=config.nuscenes_occ_voxel_size,
            nuplan_occ_voxel_size=config.nuplan_occ_voxel_size,
            occ_pillar_h=config.occ_pillar_h,
            occ_dim=config.occ_dim,
            occ_num_classes=config.occ_num_classes,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=300,
        )

    def _uvtr_voxel_to_bev_tokens(self, uvtr_voxel_space: torch.Tensor) -> torch.Tensor:
        bsz, channels, depth, bev_h, bev_w = uvtr_voxel_space.shape
        uvtr_bev_feat = self.uvtr_query_proj(uvtr_voxel_space.reshape(bsz, channels * depth, bev_h, bev_w))
        if uvtr_bev_feat.shape[0] > 1:
            uvtr_bev_feat = uvtr_bev_feat.mean(dim=0, keepdim=True)
        return uvtr_bev_feat.squeeze(0).permute(1, 2, 0).reshape(-1, uvtr_bev_feat.shape[1]).contiguous()

    def forward(self, img_vit_feats: torch.Tensor, img_llm_feats: torch.Tensor, img_metas: list[dict]):
        """Run the perception stack on the two VLM feature taps.

        ``img_vit_feats``: ``[N_cam, H, W, vit_dim]`` pre-merge patch grid.
        ``img_llm_feats``: ``[N_cam, H/2, W/2, llm_dim]`` merged image tokens.
        """
        if img_llm_feats.dim() == 5:
            B, N, H, W, C = img_llm_feats.shape
            feat_main = img_llm_feats.view(B * N, H, W, C).permute(0, 3, 1, 2)
        else:
            N, H, W, C = img_llm_feats.shape
            B = 1
            feat_main = img_llm_feats.permute(0, 3, 1, 2)

        mlvl_feats = self.adaptor(feat_main)
        mlvl_feats_reshaped = []
        for feat in mlvl_feats:
            _, C_f, H_f, W_f = feat.shape
            mlvl_feats_reshaped.append(feat.view(B, N, C_f, H_f, W_f))

        if img_vit_feats.dim() == 5:
            Bv, Nv, Hv, Wv, Cv = img_vit_feats.shape
            feat_vit = img_vit_feats.view(Bv * Nv, Hv, Wv, Cv).permute(0, 3, 1, 2)
        else:
            Nv, Hv, Wv, Cv = img_vit_feats.shape
            Bv = 1
            feat_vit = img_vit_feats.permute(0, 3, 1, 2)

        vit_mlvl_feats = self.vit_neck(feat_vit)
        vit_mlvl_feats_reshaped = []
        for vit_feat in vit_mlvl_feats:
            _, Cv_f, Hv_f, Wv_f = vit_feat.shape
            vit_mlvl_feats_reshaped.append(vit_feat.view(Bv, Nv, Cv_f, Hv_f, Wv_f))

        vit_depths = []
        for feat in vit_mlvl_feats_reshaped:
            depth_logits = self.depth_net(feat.view(-1, *feat.shape[-3:]))
            vit_depths.append(depth_logits.softmax(dim=1))

        vit_voxel_space = self.view_trans(
            vit_mlvl_feats_reshaped,
            img_depth=vit_depths,
            img_metas=copy.deepcopy(img_metas),
        )
        uvtr_occ_space = vit_voxel_space
        uvtr_bev_space = self._uvtr_voxel_to_bev_tokens(vit_voxel_space)

        return self.head(
            mlvl_feats_reshaped,
            img_metas,
            None,
            vit_bev_feat=None,
            uvtr_bev_feat=uvtr_bev_space,
            uvtr_occ_feat=uvtr_occ_space,
        )


class QwenDrivePerception(PreTrainedModel):
    """Perception head over the Qwen-Drive VLM.

    The VLM itself is not part of this checkpoint: load ``Qwen-Drive-1.0-4B``
    separately and pass its ``vlm`` module (and processor) to :meth:`attach`.
    """

    config_class = QwenDrivePerceptionConfig
    base_model_prefix = "bev_modeling"

    def __init__(self, config: QwenDrivePerceptionConfig):
        super().__init__(config)
        self.bev_modeling = BEVFormerModelV2(config)
        self.post_init()
        self._vlm = None
        self._processor = None

    def attach(self, vlm, processor) -> None:
        """Bind the VLM (e.g. ``QwenDriveForPlanning.from_pretrained(...).vlm``)."""
        self._vlm = vlm
        self._processor = processor

    # ------------------------------------------------------------------ taps

    def _modality_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids == self._vlm.config.image_token_id).long()

    @staticmethod
    def _premerge_grids(patches: torch.Tensor, image_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        """Split block-ordered pre-merge patches into per-camera ``[H, W, C]`` grids."""
        feats = []
        offset = 0
        for _, grid_h, grid_w in image_grid_thw.tolist():
            size = grid_h * grid_w
            cur = patches[offset : offset + size]
            feats.append(
                cur.view(grid_h // 2, grid_w // 2, 2, 2, cur.shape[-1])
                .permute(0, 2, 1, 3, 4)
                .reshape(grid_h, grid_w, -1)
            )
            offset += size
        return feats

    # ---------------------------------------------------------------- public

    @torch.no_grad()
    def infer(
        self,
        inputs: dict,
        img_metas: dict,
        *,
        return_latent: bool = False,
        latent_size: tuple[int, int] | None = (16, 32),
        latent_channels: int | None = 128,
    ) -> dict:
        """One perception forward: 3D boxes, occupancy and map segmentation.

        ``inputs`` is the processor output (``input_ids``, ``pixel_values``,
        ``image_grid_thw``); ``img_metas`` carries ``lidar2img``, ``lidar2ego``,
        ``img_shape``, ``dataset_type`` and ``cam_order``.

        With ``return_latent=True`` the decoded outputs are augmented with a
        spatially compressed BEV embedding. The default compression is
        ``16 x 32 x 128`` so callers can cache a compact world-model state without
        writing the full ``200 x 200 x 256`` map to disk.
        """
        if self._vlm is None:
            raise RuntimeError("call attach(vlm, processor) before infer()")

        input_ids = inputs["input_ids"]
        image_grid_thw = inputs["image_grid_thw"]

        # Capture the ViT's pre-merge patch features from the same forward the
        # language model consumes, via a hook on the patch merger.
        captured = {}

        def _hook(module, args, output=None):
            captured["patches"] = args[0]

        visual = self._vlm.model.visual
        handle = visual.merger.register_forward_hook(_hook)
        try:
            outputs = self._vlm(
                input_ids=input_ids,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=self._modality_ids(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
        finally:
            handle.remove()
        # The heads were trained on the post-norm decoder output, i.e. after the
        # language model's final norm.
        hidden_states = self._vlm.model.language_model.norm(outputs.hidden_states[-1])
        # The pre-merge tap sits after the vision tower's final norm, which lives
        # inside the merger module, so run it explicitly.
        with torch.no_grad():
            patches = visual.merger.norm(captured["patches"])
        vit_feats = self._premerge_grids(patches, image_grid_thw)

        num_cams = len(img_metas["cam_order"])
        grid_h, grid_w = image_grid_thw[-1, 1].item(), image_grid_thw[-1, 2].item()
        tokens_per_img = grid_h // 2 * grid_w // 2

        image_mask = input_ids[0] == self._vlm.config.image_token_id
        llm_tokens = hidden_states[0][image_mask]
        # Keep the trailing cameras (history frames would come first).
        llm_tokens = llm_tokens[-num_cams * tokens_per_img :]
        img_llm_feats = llm_tokens.view(num_cams, grid_h // 2, grid_w // 2, -1)
        img_vit_feats = torch.stack(vit_feats[-num_cams:], dim=0)

        dtype = next(self.bev_modeling.parameters()).dtype
        outs = self.bev_modeling(
            img_vit_feats=img_vit_feats.to(dtype),
            img_llm_feats=img_llm_feats.to(dtype),
            img_metas=[img_metas],
        )

        det = self.bev_modeling.head.get_bboxes(outs, [img_metas])[0]
        occ = self.bev_modeling.head.get_occ(outs, [img_metas])
        seg = self.bev_modeling.head.get_map_seg(outs, [img_metas])
        result = {
            "boxes": det["boxes"],
            "scores": det["scores"],
            "labels": det["labels"],
            "occ": occ[0].byte().cpu().numpy(),
            "map": seg[0].byte().cpu().numpy(),
        }
        if return_latent:
            bev_h = self.bev_modeling.head.bev_h
            bev_w = self.bev_modeling.head.bev_w
            output_hw = latent_size or (bev_h, bev_w)
            bev = _compress_feature_map(
                outs["bev_embed"],
                bev_h=bev_h,
                bev_w=bev_w,
                output_hw=output_hw,
                output_channels=latent_channels,
            )
            result["bev_embed"] = bev[0].to(torch.float16).cpu().numpy()
            result["bev_feat"] = _compress_feature_map(
                outs["bev_ego_feat_2d"],
                bev_h=bev_h,
                bev_w=bev_w,
                output_hw=output_hw,
                output_channels=latent_channels,
            )[0].to(torch.float16).cpu().numpy()
            result["latent_shape"] = tuple(bev.shape)
            result["latent_dtype"] = "float16"
        return result
