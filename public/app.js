let model
let normalizedFeature, normalizedLabel
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor
let points

const toggleVisor = () => {
  tfvis.visor().toggle()
}  

const plot = async (pointsArray, featureName, predictedPointsArray = null) => {
  const values = [pointsArray.slice(0, 1000)]
  const series = ["Original"]
  if (Array.isArray(predictedPointsArray)) {
    values.push(predictedPointsArray)
    series.push("Predicted")
  }

  tfvis.render.scatterplot(
    { name: `${featureName} vs House Price` },
    { values, series },
    {
      xLabel: featureName,
      yLabel: "Price",
    }
  )
}

const plotPredictionLine = async () => {
  const [ xs, ys ] = tf.tidy(() => {
    const normalizedXs = tf.linspace(0, 1, 100)
    const normalizedYs = model.predict(normalizedXs.reshape([100, 1]))
    const xs = denormalize(normalizedXs, normalizedFeature.min, normalizedFeature.max)
    const ys = denormalize(normalizedYs, normalizedLabel.min, normalizedLabel.max)

    return [ xs.dataSync(), ys.dataSync() ]
  })

  const predictedPoints = Array.from(xs).map((val, index) => {
    return { x: val, y: ys[index] }
  })

  await plot(points, "Square Feet", predictedPoints)
}

const train = async () => {
  // Update UI
  ["train", "test", "load", "predict", "save"].forEach(id => {
    document.getElementById(`${id}-button`).setAttribute('disabled', 'disabled')
  })
  document.getElementById('model-status').innerHTML = "Training..."

  // Fun Stuff!
  const model = createModel()
  const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor)
  const trainingLoss = result.history.loss.pop()
  const validationLoss = result.history.val_loss.pop()

  await plotPredictionLine()

  // Update UI
  document.getElementById('model-status').innerHTML = 
  `Trained (unsaved)
  Loss: ${trainingLoss.toPrecision(5)}
  Validation Loss: ${validationLoss.toPrecision(5)}`
  document.getElementById('test-button').removeAttribute('disabled')
  document.getElementById('save-button').removeAttribute('disabled')
  document.getElementById('predict-button').removeAttribute('disabled')
}

const test = async () => {
  const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor)
  const loss = await lossTensor.dataSync()

  // Update UI
  document.getElementById('testing-status').innerHTML = `Testing set loss: ${loss}`
}

const storageID = "kc-house-price-regression"
const save = async () => {
  const saveResults = await model.save(`localstorage://${storageID}`)
  document.getElementById('model-status').innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`
}

const load = async () => {
  const storageKey = `localstorage://${storageID}`
  const models = await tf.io.listModels();
  const modelInfo = models[storageKey]
  if (modelInfo) {
    model = await tf.loadLayersModel(storageKey)

    const optimizer = tf.train.adam()
  
    model.compile({
      loss: "meanSquaredError",
      optimizer,
    })

    await plotPredictionLine()

    // Update UI
    document.getElementById('model-status').innerHTML = `Trained (saved ${modelInfo.dateSaved})`
    document.getElementById('test-button').removeAttribute('disabled')
    document.getElementById('predict-button').removeAttribute('disabled')
  } else {
    alert('No Saved Model!')
  }
}

const predict = async () => {
  const predictionInput = parseInt(document.getElementById("prediction-input").value)
  if (isNaN(predictionInput)) alert("Please enter a valid number!")
  else {
    tf.tidy(() => {
      const inputTensor = tf.tensor1d([predictionInput])
      const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max)
      const normalizedOutputTensor = model.predict(normalizedInput.tensor)
      const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max)
      const outputValue = outputTensor.dataSync()[0]
      document.getElementById('prediction-output').innerHTML = `The predicted house value is $${Math.ceil(outputValue)}`
    })
  }
}

const normalize = (tensor, prevMin = null, prevMax = null) => {
  const min = prevMin || tensor.min()
  const max = prevMax || tensor.max()

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
  model = tf.sequential()

  model.add(tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'sigmoid',
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
  const { onEpochEnd } = tfvis.show.fitCallbacks({ name: "Training Performance" }, ['loss'])

  return model.fit(featureTensor, labelTensor, {
    batchSize: 32,
    epochs: 21,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd,
      onEpochBegin: async () => {
        await plotPredictionLine()
      }
    }
  })
}

const main = async () => {
  // Import and Shuffle Data
  const dataSet = await tf.data.csv('./kc_house_data.csv').toArray()
  tf.util.shuffle(dataSet)

  // Extract Useful Data and Ensure Symmetry
  points = dataSet.map(data => ({
    x: data.sqft_living,
    y: data.price
  }))
  if (points.length % 2 != 0) points.pop() 
  plot(points, "Square Feet")

  // Create Features and Labels
  const featureValues = await points.map(p => p.x)
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])
  normalizedFeature = normalize(featureTensor)

  const labelValues = await points.map(p => p.y)
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])
  normalizedLabel = normalize(labelTensor)

  // Split Data for Training and Testing
  let splitFeatureTensorArray = tf.split(normalizedFeature.tensor, 2)
  let splitLabelTensorArray = tf.split(normalizedLabel.tensor, 2)
  
  trainingFeatureTensor = splitFeatureTensorArray[0]
  testingFeatureTensor = splitFeatureTensorArray[1]
  trainingLabelTensor = splitLabelTensorArray[0]
  testingLabelTensor = splitLabelTensorArray[1]

  // Cleanup
  featureTensor.dispose()
  labelTensor.dispose()

  // Update Status and Enable Train Button
  document.getElementById("model-status").innerHTML = "No model trained"
  document.getElementById("train-button").removeAttribute('disabled')
  document.getElementById("load-button").removeAttribute('disabled')
}

main()