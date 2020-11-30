//
//  BiFPN.swift
//  
//
//  Created by Zida Wang on 11/11/20.
//

import Foundation
import TensorFlow

struct BiFPN:Layer {
    var in_channels:Array
    var out_channels:Int
    var num_ins:Int
    var num_outs:Int
    var relu_before_extra_convs:Bool
    var no_norm_on_lateral:Bool
    var stack:Int
    var backbone_end_level:Int
    var start_level:Int
    var end_level:Int
    var add_extra_convs:Bool
    var extra_convs_on_inputs:Bool
    var lateral_convs:Array
    var fpn_convs:Array
    var stack_bifpn_convs:Array
    
    
    
    init(
        in_channels:Array,
        out_channels:Array,
        num_outs:Int,
        start_level:Int = 0,
        end_level:Int = -1,
        stack:Int = 1,
        add_extra_convs:Bool = false,
        extra_convs_on_inputs:Bool = true,
        relu_before_extra_convs:Bool = false
    ) {
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = in_channels.count()
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.stack = stack
        
        if (end_level == -1) {
            self.backbone_end_level = self.num_ins
        } else {
            self.backbone_end_level = end_level
        }
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        
        for i in self.start_level...(self.backbone_end_level-1) {
            var l_conv = ConvLayer(in_channels:self.in_channels[i], out_channels: self.out_channels, kernel_size:1)
            self.lateral_convs.append(l_conv)
        }
        
        for ii in 0..<stack {
            self.stack_bifpn_convs.append(BiFPN_Layer(channels:self.out_channels, levels:self.backbone_end_level-self.start_level))
        }
        
        //add extra conv layers (e.g. RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if (self.add_extra_convs && extra_levels >= 1) {
            for i in 0..<(extra_levels) {
                if (i==0 && self.extra_convs_on_inputs) {
                    self.in_channels = self.in_channels[self.backbone_end_level-1]
                } else {
                    self.in_channels = out_channels
                }
                extra_fpn_conv = ConvLayer(in_channels:self.in_channels, self.out_channels, kernel_size:3, stride:2)
                self.fpn_convs.append(extra_fpn_conv)
            }
        }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var laterals = [Tensor<Float>]()
        for (i, lateral_conv) in self.lateral_convs.enumerated() {
            laterals.append(lateral_conv(input[i + self.start_level]))
        }
        
        var used_backbone_levels = laterals.count
        for bifpn_module in self.stack_bifpn_convs {
            laterals = bifpn_module(laterals)
        }
        var outs = laterals
        
        if self.num_outs > len(outs) {
            if !self.add_extra_convs {
                for i in 0...(self.num_outs - used_backbone_levels - 1) {
                    var maxpool = MaxPool2D<Float>(poolSize:(1,1), strides:(2,2))
                    outs.append(maxpool(outs[-1]))
                }
            } else {
                if self.extra_convs_on_inputs {
                    var orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                } else {
                    outs.append(self.fpn_convs[0](outs[-1]))
                }
                for i in 1...(self.num_outs - used_backbone_levels - 1) {
                    if self.relu_before_extra_convs {
                        outs.append(self.fpn_convs[i](relu(outs[-1])))
                    } else {
                        outs.append(self.fpn_convs[i](outs[-1]))
                    }
                }
            }
        }
        return outs
    }
}

struct BiFPN_Layer:Layer {
    var levels:Int
    var epsilon:Float
    var bifpnConvs:Array<Layer>
    var w1:Tensor<Float>
    var w2:Tensor<Float>
    var upsample:UpSampling2D<Float>
    var maxpool:MaxPool2D<Float>
    
    init(
        channels:Int,
        levels:Int,
        initial:Float=0.5,
        eps:Float = 0.0001
    ) {
        self.levels = levels
        self.epsilon = eps
        self.bifpnConvs = [ConvLayer]()
        // weighted
        self.w1 = Tensor<Float>(
            shape: TensorShape([2, levels]),
            repeating: initial)
        self.w2 = Tensor<Float>(
            shape: TensorShape([3, levels-2]),
            repeating: initial)
        
        for jj in 0...1 {
            for i in 0..<(self.levels) {
                self.bifpnConvs.append(
                    ConvLayer(
                        in_channels:channels,
                        out_channels:channels,
                        kernel_size:3,
                        padding:1))
            }
        }
        self.upsample = UpSampling2D<Float>(size:2)
        self.maxpool = MaxPool2D<Float>(poolSize:(2,2), strides:(1,1), padding: .same)
    }
    
    @differentiable
    func callAsFunction(_ inputs: Array<Tensor<Float>>) -> Tensor<Float> {
        self.w1 = relu(self.w1)
        self.w1 = self.w1 / self.w1.sum(squeezingAxes:0) + self.eps
        self.w2 = relu(self.w2)
        self.w2 = self.w2 / self.w2.sum(squeezingAxes:0) + self.eps
        
        //build top-down
        var idx_bifpn:Int = 0
        var pathtd:Array<Tensor<Float>> = inputs
        
        for i in stride(from:(self.levels-1),through:1,by:-1) {
            pathtd[i-1] = (self.w1[0,i-1] * pathtd[i-1] + self.w1[1,i-1] * self.upsample(pathtd[i])) / (self.w1[0,i-1] + self.w1[1,i-1] + self.eps)
            pathtd[i-1] = self.bifpn_convs[idx_bifpn](pathtd[i-1])
            idx_bifpn = idx_bifpn + 1
        }
        
        for i in 0...(self.levels-3) {
            pathtd[i+1] = (self.w2[0,i] * pathtd[i+1] + self.w2[1,i] * self.maxpool(pathtd[i]) + w2[2,i] * inputs[i+1]) / (self.w2[0,i] + self.w2(2,i) + self.eps)
            pathtd[i+1] = self.bifpn_convs[idx_bifpn](pathtd[i+1])
            idx_bifpn = idx_bifpn + 1
        }
        
        pathtd[self.levels-1] = (self.w1[0,self.levels-1] * pathtd[self.levels-1] + self.w1[1,self.levels-1] * self.maxpool(pathtd[self.levels-2])) / (self.w1[0,self.levels-1] + self.w1[1,self.levels-1] + self.eps)
        pathtd[self.levels-1] = self.bifpn_convs[idx_bifpn](pathtd[self.levels-1])
        
        return pathtd
    }
}
