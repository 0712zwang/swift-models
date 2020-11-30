//
//  RetinaHead.swift
//  
//
//  Created by Zida Wang on 11/11/20.
//

import Foundation
import TensorFlow

public struct RetinaHead:Layer {
    var in_channels
    var num_classes
    var feat_channels
    var anchor_scales
    var anchor_ratios
    var anchor_strides
    var stacked_convs
    var octave_base_scale
    var scales_per_octave
    var cls_out_channels
    var num_anchors
    
    var cls_convs
    var reg_convs
    var retina_cls
    
    init(
        num_classes:Int,
        in_channels:Int,
        feat_channels:Int = 256,
        anchor_scales:Array = [8, 16, 32],
        anchor_ratios:Array = [0.5, 1.0, 2.0],
        anchor_strides:Array = [4, 8, 16, 32, 64],
        stacked_convs:Int = 4,
        octave_base_scale:Int = 4,
        scales_per_octave:Int = 3
    ) {
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        
        var octave_scales = []
        for i in 0...(self.scales_per_octave.count - 1) {
            octave_scales.append(pow(2, i-scales_per_octave))
        }
        self.anchor_scales = octave_scales * octave_base_scale
        self.cls_out_channels = num_classes
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self.init_layers()
    }
    
    func init_layers() -> Void {
        self.cls_convs = []
        self.reg_convs = []
        
        for i in 0..<(self.stacked_convs) {
            var chn
            if i != 0 {
                chn = self.feat_channels
            } else {
                chn = self.in_channels
            }
            self.cls_convs.append(ConvLayer(
                                    in_channels:chn,
                                    out_channels:self.feat_channels,
                                    kernel_size:3, stride:1))
            self.reg_convs.append(ConvLayer(
                                    in_channels:chn,
                                    out_channels:self.feat_channels,
                                    kernel_size:3,
                                    stride:1))
            self.retina_cls = Conv2D<Float>(
                filterShape: (3, 3, self.feat_channels, self.num_anchors * self.cls_out_channels),
                strides: (1, 1),
                padding: .same)
            self.retina_reg = Conv2D<Float>(
                filterShape: (3, 3, self.feat_channels, self.num_anchors * self.cls_out_channels),
                strides: (1, 1),
                padding: .same)
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> (Tensor<Float>, Tensor<Float>) {
        var cls_feat = input
        var reg_feat = input
        for cls_conv in self.cls_convs {
            cls_feat = cls_conv(cls_feat)
        }
        for reg_conv in self.reg_convs {
            reg_feat = reg_conv(reg_feat)
        }
        
        var cls_score = sigmoid(self.retina_cls(cls_feat))
        //var cls_score_shape = cls_score.shape.dimensions //batch, height, width, classes
        //cls_score = cls_score.
        
        var bbox_pred = self.retina_reg(reg_feat)
        //TODO if doesn't work
        
        return (cls_score, bbox_pred)
    }
}
