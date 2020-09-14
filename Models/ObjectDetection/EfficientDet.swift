// Original Paper:
// "EfficientDet: Scalable and Efficient Object Detection"
// Mingxing Tan, Ruoming Pang, Quoc V. Le
// https://arxiv.org/pdf/1911.09070v7.pdf

import Foundation
import TensorFlow

let w_bifpns = [64, 88, 112, 160, 224, 288, 384]
let d_bifpns = [3, 4, 5, 6, 7, 7, 8]
let d_heads = [3, 3, 3, 4, 4, 4, 5]
let image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]

let MOMENTUM = 0.997
let EPSILON = 0.0001

struct SeparableConvBlock: Layer {
    var conv: SeparableConv2D<Float>
    var batchNorm: BatchNorm<Float>

    init(
        num_channels: Int,
        kernel_size: (Int, Int),
        strides: Int,
        name: String,
        freeze_bn: Bool = false
    ) {
        let filterMult = roundFilterPair(filters: filters, width: width)
        let hiddenDimension = filterMult.0 
        conv = SeparableConv2D<Float>(
            depthwiseFilter: (kernel_size.0, kernel_size.1, num_channels, 1),
            pointwiseFilter: (1, 1, num_channels, num_channels),
            strides: (strides, strides),
            padding: .same,
            bias: 
        )
        batchNorm = BatchNorm(momentum: MOMENTUM, epsilon: EPSILON)
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return swish(batchNorm(conv(input)))
    }
}


// struct ConvBlock: Layer (num_channels: (, kernel_size, strides, name, freeze_bn=False) {
//     f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
//                        use_bias=True, name='{}_conv'.format(name))
//     f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='{}_bn'.format(name))
//     f3 = ReLU(name='{}_relu'.format(name))
//     return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))
// }

// func build_BiFPN(
//     features: Tensor<Float>,
//     num_channels: Int,
//     id: Int,
//     freeze_bn: Bool = false) -> (Layer, Layer, Layer, Layer, Layer) {
//     if id == 0:
//         let P3_in = features[2]
//         let P4_in = features[3]
//         let P5_in = features[4]
//         P6_in = Conv2D(num_channels, kernel_size: 1, padding: 'same', name: 'resample_p6/conv2d')(features[4])
//         P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
//         # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
//         P6_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
//         P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
//         P7_U = UpSampling2D()(P7_in)
//         P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
//         P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
//         P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
//         P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
//                                 name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
//         P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
//                                             name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
//         # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
//         P6_U = UpSampling2D()(P6_td)
//         P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
//         P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
//         P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
//         P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
//                                 name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
//         P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
//                                             name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
//         # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
//         P5_U = UpSampling2D()(P5_td)
//         P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
//         P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
//         P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
//         P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
//                               name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
//         P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
//                                           name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
//         # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
//         P4_U = UpSampling2D()(P4_td)
//         P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
//         P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
//         P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
//         P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
//                                 name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
//         P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
//                                             name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
//         # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
//         P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
//         P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
//         P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
//         P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

//         P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
//                                 name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
//         P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
//                                             name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
//         # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
//         P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
//         P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
//         P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
//         P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

//         P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
//         P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
//         P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
//         P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

//         P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
//         P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
//         P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
//         P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

//     else:
//         P3_in, P4_in, P5_in, P6_in, P7_in = features
//         P7_U = UpSampling2D()(P7_in)
//         P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
//         P6_td = Activation(lambda x: tf.nn.swish(x))(P6_td)
//         P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
//         P6_U = UpSampling2D()(P6_td)
//         P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
//         P5_td = Activation(lambda x: tf.nn.swish(x))(P5_td)
//         P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
//         P5_U = UpSampling2D()(P5_td)
//         P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
//         P4_td = Activation(lambda x: tf.nn.swish(x))(P4_td)
//         P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
//         P4_U = UpSampling2D()(P4_td)
//         P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
//         P3_out = Activation(lambda x: tf.nn.swish(x))(P3_out)
//         P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
//         P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
//         P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
//         P4_out = Activation(lambda x: tf.nn.swish(x))(P4_out)
//         P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

//         P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
//         P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
//         P5_out = Activation(lambda x: tf.nn.swish(x))(P5_out)
//         P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

//         P5_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
//         P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
//         P6_out = Activation(lambda x: tf.nn.swish(x))(P6_out)
//         P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

//         P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
//         P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
//         P7_out = Activation(lambda x: tf.nn.swish(x))(P7_out)
//         P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
//                                     name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
//     return P3_out, P4_td, P5_td, P6_td, P7_out
// }