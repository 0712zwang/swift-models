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

import Foundation

let progressBarLength = 30

/// A handler for printing the training and validation progress. 
public class ProgressPrinter {
  /// Print training or validation progress in response of the 'event'.
  /// 
  /// An example of the progress would be:
  /// Epoch 1/12
  /// 468/468 [==============================] - loss: 0.4819 - accuracy: 0.8513
  /// 79/79 [==============================] - loss: 0.1520 - accuracy: 0.9521
  public func print<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .epochStart:
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount else {
        // No-Op if trainingLoop doesn't set the required values for progress printing.
        return
      }

      Swift.print("Epoch \(epochIndex + 1)/\(epochCount)")
    case .batchEnd:
      guard let batchIndex = loop.batchIndex, let batchCount = loop.batchCount else {
        // No-Op if trainingLoop doesn't set the required values for progress printing.
        return
      }

      let progressBar = formatProgressBar(
        progress: Float(batchIndex + 1) / Float(batchCount), length: progressBarLength)
      var stats: String = ""
      if let lastStatsLog = loop.lastStatsLog {
        stats = formatStats(lastStatsLog)
      }

      Swift.print(
        "\r\(batchIndex + 1)/\(batchCount) \(progressBar)\(stats)",
        terminator: ""
      )
      fflush(stdout)
    case .epochEnd:
      Swift.print("")
    case .validationStart:
      Swift.print("")
    default:
      return
    }
  }

  func formatProgressBar(progress: Float, length: Int) -> String {
    let progressSteps = Int(round(Float(length) * progress))
    let leading = String(repeating: "=", count: progressSteps)
    let separator: String
    let trailing: String
    if progressSteps < progressBarLength {
      separator = ">"
      trailing = String(repeating: ".", count: progressBarLength - progressSteps - 1)
    } else {
      separator = ""
      trailing = ""
    }
    return "[\(leading)\(separator)\(trailing)]"
  }

  func formatStats(_ stats: [(String, Float)]) -> String {
    var result = ""
    for stat in stats {
      result += " - \(stat.0): \(String(format: "%.4f", stat.1))"
    }
    return result
  }
}
