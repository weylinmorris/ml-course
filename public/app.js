const main = async () => {
  const normalize = (tensor) => {
    const min = tensor.min()
    const max = tensor.max()

    return {
      tensor: tensor.sub(min).div(max.sub(min)),
      min,
      max
    }
  }

  const denormalize = (tensor, min, max) => {
    return tensor.mul(max.sub(min)).add(min)
  }

  const createModel = () => {
    const model = tf.sequential()

    model.add(tf.layers.dense({
      units: 1,
      useBias: true,
      activation: 'linear',
      inputDim: 1,
    }))

    const optimizer = tf.train.adam()

    model.compile({
      loss: "meanSquaredError",
      optimizer,
    })

    return model
  }

  const trainModel = async (model, featureTensor, labelTensor) => {
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks({ name: "Training Performance" }, ['loss'])

    return model.fit(featureTensor, labelTensor, {
      batchSize: 64,
      epochs: 20,
      callbacks: {
        onEpochEnd,
      }
    })
  }

  // Import and Shuffle Data
  const dataSet = await tf.data.csv('./kc_house_data.csv').toArray()
  tf.util.shuffle(dataSet)

  // Extract Useful Data and Ensure Symmetry
  const points = dataSet.map(data => ({
    x: data.sqft_living,
    y: data.price
  }))
  if (points.length % 2 != 0) points.pop() 

  // Create Features and Labels
  const featureValues = await points.map(p => p.x)
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])
  const normalizedFeature = normalize(featureTensor)

  const labelValues = await points.map(p => p.y)
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])
  const normalizedLabel = normalize(labelTensor)

  // Split Data for Training and Testing
  const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2)
  const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2)

  // Fun Stuff!
  trainModel(createModel(), trainingFeatureTensor, trainingLabelTensor)
}

main()