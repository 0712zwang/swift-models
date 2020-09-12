import TensorFlow


/// Metrics that can be registered into TrainingLoop
public enum TrainingMetrics {
  case accuracy

  public var name: String {
  	switch self {
  	case .accuracy:
  		return "accuracy"
  	}
  }

  public var measurer: MetricsMeasurer {
  	switch self {
  	case .accuracy:
  		return AccuracyMeasurer(self.name)
  	}
  }
}

public protocol MetricsMeasurer {
	//associatedtype predictionType
	//associatedtype labelType
	//associatedtype Value
	var name: String { get set }
	mutating func accumulate(predictions: Tensor<Float>, labels: Tensor<Int32>)
	func measure() -> Float
}

public struct AccuracyMeasurer: MetricsMeasurer {
	public var name: String

	private var correctGuessCount: Int32 = 0
	private var totalGuessCount: Int32 = 0

	public init(_ name: String) {
		self.name = name
	}

	public mutating func accumulate(predictions: Tensor<Float>, labels: Tensor<Int32>) {
		correctGuessCount += Tensor<Int32>(predictions.argmax(squeezingAxis: 1) .== labels).sum().scalarized()
		totalGuessCount += Int32(labels.shape[0])
	}

	public func measure() -> Float {
		return Float(correctGuessCount) / Float(totalGuessCount)
	}
}

