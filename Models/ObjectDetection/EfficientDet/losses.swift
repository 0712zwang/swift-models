//
//  losses.swift
//  
//
//  Created by Zida Wang on 11/11/20.
//

import Foundation
import TensorFlow

public struct FocalLoss:Layer {
    init() {
        
    }
    
    @differentiable
    public func callAsFunction(
        _ classifications: Tensor<Float>,
        _ regressions: Tensor<Float>,
        _ anchors: Tensor<Float>,
        _ annotations: Tensor<Float>) -> Tensor<Float> {
    }
}
