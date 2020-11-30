//
//  modules.swift
//  
//
//  Created by Zida Wang on 11/11/20.
//

import Foundation
import TensorFlow
import PythonKit
let np = Python.import("numpy")

struct ConvLayer:Layer {
    var conv:Layer
    var norm:Layer

    init(
        in_channels:Int,
        out_channels:Int,
        kernel_size:Int,
        stride:Int=1
    ) {
        self.conv = Conv2D<Float>(
            filterShape: (kernel_size, kernel_size, in_channels, out_channels),
            strides: (stride, stride),
            padding: .same)
        self.norm = BatchNorm(featureCount:out_channels)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return relu(norm(conv(input)))
    }
}

public struct RegressionModel:Layer {
    var conv1:Conv2D<Float>
    var conv2:Conv2D<Float>
    var conv3:Conv2D<Float>
    var conv4:Conv2D<Float>
    var output:Conv2D<Float>
    
    init(
        num_features_in:Int,
        num_anchors:Int=9,
        feature_size:Int=256
    ) {
        self.conv1 = Conv2D<Float>(filterShape: (3,3,num_features_in, feature_size), padding=.same)
        self.conv2 = Conv2D<Float>(filterShape: (3,3,feature_size, feature_size), padding=.same)
        self.conv3 = Conv2D<Float>(filterShape: (3,3,feature_size, feature_size), padding=.same)
        self.conv4 = Conv2D<Float>(filterShape: (3,3,feature_size, feature_size), padding=.same)
        self.output = Conv2D<Float>(filterShape: (3,3,feature_size, num_anchors*4), padding=.same)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        out = self.conv1(x)
        out = relu(out)
        out = self.conv2(out)
        out = relu(out)
        out = self.conv3(out)
        out = relu(out)
        out = self.conv4(out)
        out = relu(out)
        out = self.output(out)
        // out is B x C x W x H, with C = 4*num_anchors
        // out = out.permute(0, 2, 3, 1)
        return out //.contiguous().view(out.shape[0], -1, 4)
    }
}

public struct ClassificationModel:Layer {'
    var conv1:Conv2D<Float>
    var conv2:Conv2D<Float>
    var conv3:Conv2D<Float>
    var conv4:Conv2D<Float>
    var output:Conv2D<Float>
    var num_classes:Int
    var num_anchors:Int
    
    init(
        num_features_in:Int,
        num_anchors:Int=9,
        num_classes:Int=80,
        prior:Float=0.01,
        feature_size:Int=256
    ) {
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = Conv2D<Float>(filterShape: (3,3,num_features_in, feature_size), padding=.same)
        self.conv2 = Conv2D<Float>(filterShape: (3,3,feature_size, feature_size), padding=.same)
        self.conv3 = Conv2D<Float>(filterShape: (3,3,feature_size, feature_size), padding=.same)
        self.conv4 = Conv2D<Float>(filterShape: (3,3,feature_size, feature_size), padding=.same)
        self.output = Conv2D<Float>(filterShape: (3,3,feature_size, num_anchors*num_classes), padding=.same)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        out = self.conv1(x)
        out = relu(out)
        out = self.conv2(out)
        out = relu(out)
        out = self.conv3(out)
        out = relu(out)
        out = self.conv4(out)
        out = relu(out)
        out = self.output(out)
        out = sigmoid(out)
        return out
    }
}

public struct Anchors:Layer {
    var pyramid_levels
    var strides
    var sizes
    var ratios
    var scales
    
    init(
        pyramid_levels:Array=nil,
        strides:Array=nil,
        sizes:Array=nil,
        ratios:PythonObject=nil,
        scales:PythonObject=nil
    ) {
        if pyramid_levels == nil {
            self.pyramid_levels = [3, 4, 5, 6, 7]
        }
        if strides == nil {
            self.strides = self.pyramid_levels.map { pow(2, $0) }
        }
        if sizes == nil {
            self.sizes = self.pyramid_levels.map { pow(2, $0+2) }
        }
        if ratios == nil {
            self.ratios = np.array([0.5, 1.0, 2.0])
        }
        if scales == nil {
            self.scales = np.array([pow(2.0,0.0), pow(2.0,1.0/3.0), pow(2.0,2.0/3.0)])
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var image_shape = np.array(input.shape[2...])
        var image_shapes = [PythonObject]()
        for x in self.pyramid_levels {
            image_shapes.append((image_shape + pow(2,x) - 1) / pow(2,x))
        }
        
        var all_anchors = np.zeros((0, 4)).astype(np.float32)
        
        for (idx, p) in zip(self.pyramid_levels.indices, self.pyramid_levels) {
            var anchors = generate_anchors(base_size:self.sizes[idx], ratios:self.ratios, scales:self.scales)
            var shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        }
        return Tensor<Float>(all_anchors)
    }
}

func generate_anchors(base_size:Int=16, ratios:Array=nil, scales:Array=nil) -> Array {
    if ratios == nil {
        ratios = np.array([0.5, 1, 2])
    }
    if scales == nil {
        scales = np.array([pow(2.0,0), pow(2.0,1.0/3.0), pow(2.0,2.0/3.0)])
    }
    var num_anchors = ratios.count * scales.count
    
    var anchors = np.zeros((num_anchors, 4))
    
    //scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    
    //compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    
    //correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, scales.count))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, scales.count))
    
    //transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    
    return anchors
}

func shift(shape:PythonObject, stride:Int, anchors:PythonObject) {
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    // add A anchors (1, A, 4) to
    // cell K shifts (K, 1, 4) to get
    // shift anchors (K, A, 4)
    // reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors
}

public struct BBoxTransform:Layer {
    var mean:Array
    var std:Array
    
    init(
        mean:Array=nil,
        std:Array=nil
    ) {
        if mean == nil {
            self.mean = Tensor<Float>([0,0,0,0])
        } else {
            self.mean = mean
        }
        if std == nil {
            self.std = Tensor<Float>([0.1, 0.1, 0.2, 0.2])
        } else {
            self.std = std
        }
    }
    
    //TODO
    @differentiable
    public func callAsFunction(_ boxes: Tensor<Float>, _ deltas: Tensor<Float>) -> Tensor<Float> {
    }
}

//TODO
public struct ClipBoxes:Layer {
    init() {
        
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    }
}
