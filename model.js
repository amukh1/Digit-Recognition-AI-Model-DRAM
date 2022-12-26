var math = require('mathjs')
var mnist = require('mnist').set(8000, 2000);

console.log("starting AI...")

var testV = 0
var input = mnist.training[testV].input
var output = mnist.training[testV].output // correct answer

let functions = {
  ReLU: function ReLU(x) {
    if (x[0] < 0) return 0; else if (x[0] > 0) return x[0]
  },
  softmax: function softmax(arr) {
    return arr.map(function(value, index) {
      return Math.exp(value) / arr.map(function(y /*value*/) { return Math.exp(y) }).reduce(function(a, b) { return a + b })
    })
  }, none: (x) => x
}

let WeightsAndBias = [
  [math.random([10, 784], -0.5, 0.5), math.random([10, 1], -0.5, 0.5)],
  [math.random([10, 10], -0.5, 0.5), math.random([10, 1], -0.5, 0.5)],
  [math.random([10, 10], -0.5, 0.5), math.random([10, 1], -0.5, 0.5)]
]

let op = l0(input)
console.log(op) // AI response
console.log(output) // Correct response
let error = math.multiply(math.subtract(op.output, output),math.subtract(op.output, output))
console.log(error)

function l0(I) /* I = [784x1] */  {
  let W = WeightsAndBias[0][0]
  let B = WeightsAndBias[0][1]

  let output = math.add(math.transpose([math.multiply(W, I)]), B)
  
  let outputReLU = output.map((x) => {
    return functions.ReLU(x)
  })
  let o = l1(outputReLU)
   return {output:o.output, l0: [output, outputReLU], l1:o.l1, l2:o.l2}
}

function l1(I) /* I = [10x1] */ {
  let W1 = WeightsAndBias[1][0]
  let B1 = WeightsAndBias[1][1]

  let output = math.add(math.transpose([math.multiply(W1, I)]), B1)

  let outputReLU = output.map((x) => {
    return functions.ReLU(x)
  })
  let o = l2(outputReLU)
  return {output:o.output, l1: [output,outputReLU], l2:o.l2}
}

function l2(I) /* I = [10x1] */ {
  let W2 = WeightsAndBias[2][0]
  let B2 = WeightsAndBias[2][1]

  let output = math.add(math.transpose([math.multiply(W2, I)]), B2)

  let outputSM = functions.softmax(output)
    return {output:outputSM, l2: [output,outputSM]}
}

function backprop(oData, Y) {
 
}

function gradient() {
  
}
