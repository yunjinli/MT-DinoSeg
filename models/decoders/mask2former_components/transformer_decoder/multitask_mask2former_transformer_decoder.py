from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from typing import List
import torch
from torch import nn

class MultitaskMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(self, dataset_prediction_mapping, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # num_queries = 0
        # for key, value in num_obj_queries_dict.items():
        #     num_queries += value
        # assert num_queries == self.num_queries, f"Total number of object queries {num_obj_queries_dict} !=  {self.num_queries}"

        # self.obj_queries_start_end_indices = {}
        # index = 0
        # for key, value in num_obj_queries_dict.items():
        #     start_index = index
        #     index += value
        #     end_index = index
        #     self.obj_queries_start_end_indices[key] = [start_index, end_index]
        
        # self.class_logits_start_end_indices = {}
        # index = 0
        # for key, value in num_class_dict.items():
        #     start_index = index
        #     index += value
        #     end_index = index
        #     self.class_logits_start_end_indices[key] = [start_index, end_index]
        self.dataset_prediction_mapping = {}
        for key, value in dataset_prediction_mapping.items():
            self.dataset_prediction_mapping[key] = torch.from_numpy(value.T)

    def decompose_task_specific_outputs(self, outputs):
        ## Output format
        # out = {
        #     'pred_logits': predictions_class[-1], shape: [B, Q, C]
        #     'pred_masks': predictions_mask[-1], shape: [B, Q, H, W]
        #     'aux_outputs': self._set_aux_loss(
        #         predictions_class if self.mask_classification else None, predictions_mask
        #     )
        # }
        all_pred_logits = outputs['pred_logits'] # shape: [B, Q, C]
        all_pred_masks = outputs['pred_masks'] # shape: [B, Q, H, W]
        all_aux_outputs = outputs['aux_outputs']
        
        # task_pred_logits = {}
        # task_pred_masks = {}
        # task_aux_outputs = {}

        task_outputs = {}

        for key, value in self.dataset_prediction_mapping.items():
            task_outputs[key] = {}
            task_outputs[key]['aux_outputs'] = []
            task_outputs[key]['pred_logits'] = torch.cat([all_pred_logits[:, :, :-1] @ value.to(all_pred_logits.device), all_pred_logits[:, :, -1:]], dim=-1)
            task_outputs[key]['pred_masks'] = all_pred_masks
            
            for all_aux_output in all_aux_outputs:
                a = torch.cat([all_aux_output["pred_logits"][:, :, :-1] @ value.to(all_pred_logits.device), all_aux_output["pred_logits"][:, :, -1:]], dim=-1)
                b = all_aux_output["pred_masks"]
                task_outputs[key]['aux_outputs'].append({"pred_logits": a, "pred_masks": b})
        # for key, value in self.obj_queries_start_end_indices.items():
        #     task_outputs[key] = {}
        #     task_outputs[key]['pred_logits'] = torch.cat([all_pred_logits[:, value[0]:value[1], self.class_logits_start_end_indices[key][0]:self.class_logits_start_end_indices[key][1]], 
        #                                     all_pred_logits[:, value[0]:value[1], -1:]],
        #                                     dim=-1
        #                                 )
        #     task_outputs[key]['pred_masks'] = all_pred_masks[:, value[0]:value[1], :, :]

        #     task_outputs[key]['aux_outputs'] = []

        #     for all_aux_output in all_aux_outputs:
        #         a = all_aux_output["pred_logits"]
        #         b = all_aux_output["pred_masks"]

        #         aa = torch.cat([a[:, value[0]:value[1], self.class_logits_start_end_indices[key][0]:self.class_logits_start_end_indices[key][1]], 
        #                                     a[:, value[0]:value[1], -1:]],
        #                                     dim=-1
        #                                 )
        #         bb = b[:, value[0]:value[1], :, :]
        #         task_outputs[key]['aux_outputs'].append({"pred_logits": aa, "pred_masks": bb})
        
        return outputs, task_outputs
        
        


            

