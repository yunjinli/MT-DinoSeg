import torch
import torch.nn as nn
from torch.nn import functional as F

from config.base import DecoderConfig
from .base import BaseDecoder
from .mask2former_components import MSDeformAttnPixelDecoder, MultiScaleMaskedTransformerDecoder, OpenMultiScaleMaskedTransformerDecoder, MultitaskMultiScaleMaskedTransformerDecoder

class Mask2FormerHead(BaseDecoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        self.mask_classification = getattr(config, 'mask_classification', True)
        if not hasattr(config, 'input_shape') or not config.input_shape:
            raise ValueError("Mask2Former requires input_shape to be specified in config")

        input_shape = {}
        for name, shape_info in config.input_shape.items():
            if isinstance(shape_info, dict):
                # Convert dict to our ShapeSpec-like object
                input_shape[name] = type('ShapeSpec', (), shape_info)
            else:
                input_shape[name] = shape_info
        
        self.pixel_decoder = self._create_pixel_decoder(input_shape, config)
        self.predictor = self._create_transformer_decoder(config)
        self.transformer_in_feature = getattr(config, 'maskformer_in_feature', 'multi_scale_pixel_decoder')

    def _create_pixel_decoder(self, input_shape, config):
        """
        Create the pixel decoder component.
        """
        # Try to create the MSDeformAttnPixelDecoder with our configuration
        return MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=getattr(config, 'transformer_dropout', 0.0),
            transformer_nheads=getattr(config, 'transformer_nheads', 8),
            transformer_dim_feedforward=getattr(config, 'transformer_dim_feedforward', 1024),
            transformer_enc_layers=getattr(config, 'transformer_enc_layers', 6),
            conv_dim=getattr(config, 'conv_dim', 256),
            mask_dim=getattr(config, 'mask_dim', 256),
            norm=getattr(config, 'norm', 'GN'),
            transformer_in_features=getattr(config, 'transformer_in_features', list(input_shape.keys())),
            common_stride=getattr(config, 'common_stride', 4),
        )
    
    def _create_transformer_decoder(self, config):
        """
        Create the transformer decoder component.
        """
        # Determine input channels for the transformer decoder
        conv_dim = getattr(config, 'conv_dim', 256)
        
        return MultiScaleMaskedTransformerDecoder(
            in_channels=conv_dim,
            mask_classification=self.mask_classification,
            num_classes=config.num_classes,
            hidden_dim=getattr(config, 'maskformer_hidden_dim', 256),
            num_queries=getattr(config, 'num_obj_queries', 100),
            nheads=getattr(config, 'maskformer_nheads', 8),
            dim_feedforward=getattr(config, 'maskformer_dim_feedforward', 2048),
            dec_layers=getattr(config, 'dec_layers', 9),
            pre_norm=getattr(config, 'pre_norm', False),
            mask_dim=getattr(config, 'mask_dim', 256),
            enforce_input_project=getattr(config, 'enforce_input_project', False),
        )

    def forward(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert len(inputs) == len(self.config.input_shape)
            features = {}
            keys = list(self.config.input_shape.keys())
            for i in range(len(inputs)):
                features[keys[i]] = inputs[i]
                # print(f"{keys[i]}: {features[keys[i]].shape}")
            inputs = features
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features=inputs)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(inputs[self.transformer_in_feature], mask_features, mask)
        
        return predictions

    def get_num_queries(self):
        """Get the number of object queries used by this model."""
        return getattr(self.config, 'num_obj_queries', 100)
    
    def get_mask_dim(self):
        """Get the mask feature dimension."""
        return getattr(self.config, 'mask_dim', 256)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

class MultitaskMask2FormerHead(BaseDecoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        self.mask_classification = getattr(config, 'mask_classification', True)
        if not hasattr(config, 'input_shape') or not config.input_shape:
            raise ValueError("Mask2Former requires input_shape to be specified in config")

        input_shape = {}
        for name, shape_info in config.input_shape.items():
            if isinstance(shape_info, dict):
                # Convert dict to our ShapeSpec-like object
                input_shape[name] = type('ShapeSpec', (), shape_info)
            else:
                input_shape[name] = shape_info
        
        self.pixel_decoder = self._create_pixel_decoder(input_shape, config)
        self.predictor = self._create_transformer_decoder(config)
        self.transformer_in_feature = getattr(config, 'maskformer_in_feature', 'multi_scale_pixel_decoder')

    def _create_pixel_decoder(self, input_shape, config):
        """
        Create the pixel decoder component.
        """
        # Try to create the MSDeformAttnPixelDecoder with our configuration
        return MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=getattr(config, 'transformer_dropout', 0.0),
            transformer_nheads=getattr(config, 'transformer_nheads', 8),
            transformer_dim_feedforward=getattr(config, 'transformer_dim_feedforward', 1024),
            transformer_enc_layers=getattr(config, 'transformer_enc_layers', 6),
            conv_dim=getattr(config, 'conv_dim', 256),
            mask_dim=getattr(config, 'mask_dim', 256),
            norm=getattr(config, 'norm', 'GN'),
            transformer_in_features=getattr(config, 'transformer_in_features', list(input_shape.keys())),
            common_stride=getattr(config, 'common_stride', 4),
        )
    
    def _create_transformer_decoder(self, config):
        """
        Create the transformer decoder component.
        """
        # Determine input channels for the transformer decoder
        conv_dim = getattr(config, 'conv_dim', 256)
        
        return MultitaskMultiScaleMaskedTransformerDecoder(
            num_obj_queries_dict=config.num_obj_queries_dict, 
            num_class_dict=config.num_class_dict,
            in_channels=conv_dim,
            mask_classification=self.mask_classification,
            num_classes=config.num_classes,
            hidden_dim=getattr(config, 'maskformer_hidden_dim', 256),
            num_queries=getattr(config, 'num_obj_queries', 100),
            nheads=getattr(config, 'maskformer_nheads', 8),
            dim_feedforward=getattr(config, 'maskformer_dim_feedforward', 2048),
            dec_layers=getattr(config, 'dec_layers', 9),
            pre_norm=getattr(config, 'pre_norm', False),
            mask_dim=getattr(config, 'mask_dim', 256),
            enforce_input_project=getattr(config, 'enforce_input_project', False),
        )

    def forward(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert len(inputs) == len(self.config.input_shape)
            features = {}
            keys = list(self.config.input_shape.keys())
            for i in range(len(inputs)):
                features[keys[i]] = inputs[i]
                # print(f"{keys[i]}: {features[keys[i]].shape}")
            inputs = features
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features=inputs)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(inputs[self.transformer_in_feature], mask_features, mask)
        
        predictions = self.predictor.decompose_task_specific_outputs(outputs=predictions)

        return predictions

    def get_num_queries(self):
        """Get the number of object queries used by this model."""
        return getattr(self.config, 'num_obj_queries', 100)
    
    def get_mask_dim(self):
        """Get the mask feature dimension."""
        return getattr(self.config, 'mask_dim', 256)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

class OpenMask2FormerHead(BaseDecoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        self.mask_classification = getattr(config, 'mask_classification', True)
        if not hasattr(config, 'input_shape') or not config.input_shape:
            raise ValueError("Mask2Former requires input_shape to be specified in config")

        input_shape = {}
        for name, shape_info in config.input_shape.items():
            if isinstance(shape_info, dict):
                # Convert dict to our ShapeSpec-like object
                input_shape[name] = type('ShapeSpec', (), shape_info)
            else:
                input_shape[name] = shape_info
        
        self.pixel_decoder = self._create_pixel_decoder(input_shape, config)
        self.predictor = self._create_transformer_decoder(config)
        self.transformer_in_feature = getattr(config, 'maskformer_in_feature', 'multi_scale_pixel_decoder')

    def _create_pixel_decoder(self, input_shape, config):
        """
        Create the pixel decoder component.
        """
        # Try to create the MSDeformAttnPixelDecoder with our configuration
        return MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=getattr(config, 'transformer_dropout', 0.0),
            transformer_nheads=getattr(config, 'transformer_nheads', 8),
            transformer_dim_feedforward=getattr(config, 'transformer_dim_feedforward', 1024),
            transformer_enc_layers=getattr(config, 'transformer_enc_layers', 6),
            conv_dim=getattr(config, 'conv_dim', 256),
            mask_dim=getattr(config, 'mask_dim', 256),
            norm=getattr(config, 'norm', 'GN'),
            transformer_in_features=getattr(config, 'transformer_in_features', list(input_shape.keys())),
            common_stride=getattr(config, 'common_stride', 4),
        )
    
    def _create_transformer_decoder(self, config):
        """
        Create the transformer decoder component.
        """
        # Determine input channels for the transformer decoder
        conv_dim = getattr(config, 'conv_dim', 256)
        
        return OpenMultiScaleMaskedTransformerDecoder(
            class_names=config.class_names,
            in_channels=conv_dim,
            mask_classification=self.mask_classification,
            num_classes=config.num_classes,
            hidden_dim=getattr(config, 'maskformer_hidden_dim', 256),
            num_queries=getattr(config, 'num_obj_queries', 100),
            nheads=getattr(config, 'maskformer_nheads', 8),
            dim_feedforward=getattr(config, 'maskformer_dim_feedforward', 2048),
            dec_layers=getattr(config, 'dec_layers', 9),
            pre_norm=getattr(config, 'pre_norm', False),
            mask_dim=getattr(config, 'mask_dim', 256),
            enforce_input_project=getattr(config, 'enforce_input_project', False),
        )

    def forward(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert len(inputs) == len(self.config.input_shape)
            features = {}
            keys = list(self.config.input_shape.keys())
            for i in range(len(inputs)):
                features[keys[i]] = inputs[i]
                # print(f"{keys[i]}: {features[keys[i]].shape}")
            inputs = features
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features=inputs)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(inputs[self.transformer_in_feature], mask_features, mask)
        
        return predictions

    def get_num_queries(self):
        """Get the number of object queries used by this model."""
        return getattr(self.config, 'num_obj_queries', 100)
    
    def get_mask_dim(self):
        """Get the mask feature dimension."""
        return getattr(self.config, 'mask_dim', 256)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

