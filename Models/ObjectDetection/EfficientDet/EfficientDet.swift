// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

// Original Paper:
// "EfficientDet: Scalable and Efficient Object Detection"
// Mingxing Tan, Ruoming Pang, Quoc V. Le
// https://arxiv.org/pdf/1911.09070v7.pdf

public struct EfficientDet: Layer {

    var backbone: EfficientNet
    var neck:BiFPN
    var bbox_head:RetinaHead

    // default settings are efficientdetD0 (baseline) network
    public init(
        num_classes:Int=5,
        network:String='efficientdet-d0',
        D_bifpn:Int=3,
        W_bifpn:Int=64,
        D_class:Int=3,
        is_training:Bool=false,
        threshold=0.01,
        iou_threshold=0.5
    ) {
        
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
    }
}

extension EfficientDet {
    public enum Kind {
        case efficientdetD0
        case efficientdetD1
        case efficientdetD2
        case efficientdetD3
        case efficientdetD4
        case efficientdetD5
        case efficientdetD6
        case efficientdetD7
        case efficientdetD7x
    }

    public init(kind: Kind, classCount: Int = 1000) {
        switch kind {
        case .efficientdetD0:
            self.init(network:'efficientdet-d0', D_bifpn=3, W_bifpn=64, D_class=3, is_training=false)
        case .efficientdetD1:
            self.init(network:'efficientdet-d1', D_bifpn=4, W_bifpn=88, D_class=3, is_training=false)
        case .efficientdetD2:
            self.init(network:'efficientdet-d2', D_bifpn=5, W_bifpn=112, D_class=3, is_training=false)
        case .efficientdetD3:
            self.init(network:'efficientdet-d3', D_bifpn=6, W_bifpn=160, D_class=4, is_training=false)
        case .efficientdetD4:
            self.init(network:'efficientdet-d4', D_bifpn=7, W_bifpn=224, D_class=4, is_training=false)
        case .efficientdetD5:
            self.init(network:'efficientdet-d5', D_bifpn=7, W_bifpn=288, D_class=4, is_training=false)
        case .efficientdetD6:
            self.init(network:'efficientdet-d6', D_bifpn=8, W_bifpn=384, D_class=5, is_training=false)
        case .efficientdetD7:
            self.init(network:'efficientdet-d7', D_bifpn=8, W_bifpn=384, D_class=5, is_training=false)
        case .efficientdetD7x:
            self.init(network:'efficientdet-d7x', D_bifpn=8, W_bifpn=384, D_class=5, is_training=false)
        }
    }
}
