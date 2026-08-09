# Copyright 2021 RangiLyu.
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

import copy

import torch

from ..head import build_head
from .one_stage_detector import OneStageDetector


class NanoDetPlus(OneStageDetector):
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
    ):
        super(NanoDetPlus, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    # In nanodet/model/arch/nanodet_plus.py
    
    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        if self.epoch >= self.detach_epoch:
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = tuple(
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = tuple(
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
    
        # --- START: MODIFIED LOGIC ---
    
        # 1. Get main head output and calculate its loss (Sigmoid/QFL)
        head_out = self.head(fpn_feat)
        loss, loss_states = self.head.loss(head_out, gt_meta)
    
        # 2. Get aux head output and calculate its loss (Softmax/CrossEntropy)
        aux_head_out = self.aux_head(dual_fpn_feat)
        # Call the aux_head's OWN loss method
        aux_loss, aux_loss_states = self.aux_head.loss(aux_head_out, gt_meta)
    
        # 3. Combine the losses for backpropagation
        loss = loss + aux_loss
        for k, v in aux_loss_states.items():
            loss_states["aux_" + k] = v
            
        # --- END: MODIFIED LOGIC ---
            
        return head_out, loss, loss_states

    def inference(self, meta):
        """Run inference with the aux head active so that softmax scores are
        attached to every detection (mirrors ``get_full_scores.py``)."""
        with torch.no_grad():
            feat = self.backbone(meta["img"])
            fpn_feat = self.fpn(feat)
            preds = self.head(fpn_feat)

            # Run the auxiliary head to obtain per-detection softmax scores.
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = tuple(
                torch.cat([f, aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
            aux_preds = self.aux_head(dual_fpn_feat)

        results = self.head.post_process(preds, meta, aux_preds=aux_preds)
        return results


