// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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


/// A callback-based handler of statistics obtained during a training loop. This can be employed
/// by progress bars, recorders, or logging functionality.
public class StatisticsRecorder {
  public var liveStatistics: Bool

  var metricMeasurers: [MetricsMeasurer]
  var totalBatchLoss: Float = 0
  var batchCount: Int32 = 0

  /// Initializes the statistics tracker with
  ///
  /// - Parameters:
  ///   - metrics: A set of TrainingMetrics to capture during the training loop.
  public init(liveStatistics: Bool = true, metrics: [TrainingMetrics]) {
    self.liveStatistics = liveStatistics
    metricMeasurers = metrics.map { $0.measurer }
  }

  /// The callback used to hook into the TrainingLoop. This is updated once per event.
  ///
  /// - Parameters:
  ///   - loop: The TrainingLoop where an event has occurred. This can be accessed to obtain
  ///     the last measure loss and other values.
  ///   - event: The training or validation event that this callback is responding to.
  public func record<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .trainingStart, .validationStart:
      totalBatchLoss = 0
      batchCount = 0
    case .batchEnd:
      if let lastStepLoss = loop.lastStepLoss {
        accumulateLoss(lastStepLoss.scalarized())
      }
      guard let possibleOutput = loop.lastStepOutput, 
            let possibleTarget = loop.lastStepTarget else { return }
      guard let output = possibleOutput as? Tensor<Float>,
            let target = possibleTarget as? Tensor<Int32> else {
        fatalError(
          "For accuracy measurements, the model output must be Tensor<Float>, and the labels must be Tensor<Int>.")
      }
      accumulateMetrics(predictions: output, labels: target)

      guard let batchIndex = loop.batchIndex, let batchCount = loop.batchCount else {
        return
      }
      if liveStatistics || (batchCount == (batchIndex + 1)) {
        loop.lastStatsLog = computeStats()
      }
    default:
      return
    }
  }

  func accumulateLoss(_ newBatchLoss: Float) {
    totalBatchLoss += newBatchLoss
    batchCount += 1
  }

  func accumulateMetrics(predictions: Tensor<Float>, labels: Tensor<Int32>) {
    metricMeasurers = metricMeasurers.map { (measurer) -> MetricsMeasurer in
      var measurer = measurer
      measurer.accumulate(predictions: predictions, labels: labels) 
      return measurer
    }
  }

  func computeStats() -> [(String, Float)] {
    return [("loss", computeLoss())] + computeMetrics()
  }

  func computeLoss() -> Float {
    return totalBatchLoss / Float(batchCount)
  }

  func computeMetrics() -> [(String, Float)] {
    var result: [(String, Float)] = []
    for measurer in metricMeasurers {
      result.append((measurer.name, measurer.measure()))
    }
    return result
  }
}
