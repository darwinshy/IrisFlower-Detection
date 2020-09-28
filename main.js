const tf = require("@tensorflow/tfjs");
const iris = require("./iris.json");
const irisTesting = require("./iris-testing.json");

//[[[Setting up the Data-Set]]]

// Note:    Converting each item to a Flat 2D array
//          so that it can be used to train the model
const trainingData = tf.tensor2d(
  iris.map((item) => [
    item.sepal_length,
    item.sepal_width,
    item.petal_length,
    item.petal_width,
  ])
);
const outputData = tf.tensor2d(
  iris.map((item) => [
    item.species === "setosa" ? 1 : 0,
    item.species === "virginica" ? 1 : 0,
    item.species === "versicolor" ? 1 : 0,
  ])
);
const testingData = tf.tensor2d(
  irisTesting.map((item) => [
    item.sepal_length,
    item.sepal_width,
    item.petal_length,
    item.petal_width,
  ])
);

//[[[Building a Neural Network]]]

// Note:    A sequential model is any model
//          where the outputs of one layer are
//          the inputs to the next layer,
//          i.e. the model topology is a simple
//          stack of layers, with no branching or skipping.

const model = tf.sequential();

// Note:    Sigmoid works best classification

// Layer 1, Input is 4 and Output is 5
model.add(
  tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5,
  })
);
// Layer 2, Input is 5 and Output is 3
model.add(
  tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3,
  })
);
// Layer 3, Input is dynamic and Output is 3
model.add(
  tf.layers.dense({
    activation: "sigmoid",
    units: 3,
  })
);

// [[[Error Handling and Compiling]]]
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(0.06),
});

// [[[Training the model and Fitting our Networks]]]

//Note:     Epochs determines the number of time
//          the data set has to be trained.

const stTime = Date.now();
// Training our model using fit method
var trainedModel = model.fit(trainingData, outputData, { epochs: 100 });
const ftTime = Date.now();

// [[[Predicting]]]

// Note:    Using the trained model to predict a certain data-set.
//          For this case we use iris-testing.json

trainedModel.then((result) => {
  console.log(`Time Duration : ${ftTime - stTime} seconds`);

  model.predict(testingData).print();
});
