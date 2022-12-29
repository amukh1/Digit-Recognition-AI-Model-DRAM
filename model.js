var math = require('mathjs')
var mnist = require('mnist').set(8000, 2000);
// let crossEntropy = require("@tensorflow/tfjs").losses.sigmoidCrossEntropy

console.log("starting AI...")

var testV = 0
var input = mnist.training[testV].input
var output = mnist.training[testV].output // correct answer

let functions = {
  ReLU: function ReLU(ix) {
    // console.log(ix._data.map(x=>(x+1)))
    return ix._data.map(x=>{
      if (x < 0) return 0; else if (x > 0) return x
    })
  },
  softmax: function softmax(arr) {
    // return arr.map(function(value, index) {
    //   return Math.exp(value) / arr.map(function(y /*value*/) { return Math.exp(y) }).reduce(function(a, b) { return a + b })
    // })
    // console.log('arr',arr)
    return arr._data.map((x)=>{
      // console.log(x)
      return 1/(1+Math.pow(Math.E,-x))
    })
  },
  crossEntropy: (yh,k)=>{
    return yh.map((yhi,i)=>(Math.log(yhi)*k[i]))
  },
  arrSum: (x)=>{let r = 0; x.forEach((ih1)=>{r+=ih1}); return r},
  none: (x) => x
}

let WaB = [
  [math.random([10,784], -0.5, 0.5), math.random(-0.5, 0.5)],
  [math.random([10, 10], -0.5, 0.5), math.random(-0.5, 0.5)],
  [math.random([10, 10], -0.5, 0.5), math.random(-0.5, 0.5)]
]

let op2 = run(input, WaB)
console.log('out:', op2.output.map(x=>(Math.round(100*x)))) // AI response
console.log(output.map(x=>(Math.round(100*x)))) // Correct response
console.log(-functions.arrSum(functions.crossEntropy(op2.output, output)))

function run(Ie, WeightsAndBias) {
 
  let op = l0(Ie)
return op

// let WeightsAndBias = [
//   [math.random([10,784], -0.5, 0.5), math.random(-0.5, 0.5)],
//   [math.random([10, 10], -0.5, 0.5), math.random(-0.5, 0.5)],
//   [math.random([10, 10], -0.5, 0.5), math.random(-0.5, 0.5)]
// ]


// let error = math.multiply(math.subtract(op.output, output),math.subtract(op.output, output))
// console.log(error)

// let mvcAI = (I, w1, w2, w3, b1, b2, b3) => functions.softmax(math.add(math.transpose(math.multiply(w3,functions.ReLU(math.add(math.transpose(math.multiply(w2,functions.ReLU(math.add(math.transpose(math.multiply(w1, I)),b1)))),b2)))),b3))
// functions.softmax(math.add(math.multiply(w3,functions.ReLU(math.add(math.mutliply(w2,functions.ReLU(math.add(math.multiply(w1, I),b1))),b2))),b3))
// (I, w1, w2, w3, b1, b2, b3) => (functions.softmax(w3*functions.ReLU(w2*functions.ReLU(w1*(I)+b1)+b2)+b3))

// console.log(mvcAI(input, WeightsAndBias[0][0],WeightsAndBias[1][0],WeightsAndBias[2][0],WeightsAndBias[0][1],WeightsAndBias[1][1],WeightsAndBias[2][1]))

function neuron(I, W, B, i) {
   let sig = 0
   for(var i1 = 0; i1 < W.length; i1++){
     sig += I[i1]*W[i1]
   }
  // console.log(sig)
   return sig + B
}

function l0(I) /* I = [784x1] */  {
  let W = WeightsAndBias[0][0]
  let B = WeightsAndBias[0][1]

  // let output = math.add(([math.multiply(W, (I))]), B)
  let output = math.zeros(10).map((x,i)=>{
    return neuron(I, W[i], B, i)
  })


  let outputReLU = functions.ReLU(output)
      // console.log(outputReLU)
      console.log('l1 passed')
  let o = l1(outputReLU)
   return {output:o.output, l0: [output, outputReLU], l1:o.l1, l2:o.l2}
}

function l1(I) /* I = [10x1] */ {
  let W = WeightsAndBias[1][0]
  let B = WeightsAndBias[1][1]

  // let output = math.add((math.dot(W, I)), B)
let output = math.zeros(10).map((x,i)=>{
    return neuron(I, W[i], B, i)
  })
  
  let outputReLU = functions.ReLU(output)
  let o = l2(outputReLU)
  return {output:o.output, l1: [output,outputReLU], l2:o.l2}
}

function l2(I) /* I = [10x1] */ {
  let W = WeightsAndBias[2][0]
  let B = WeightsAndBias[2][1]

  // let output = math.add(math.transpose([math.multiply(W2, I)]), B2)
let output = math.zeros(10).map((x,i)=>{
    return neuron(I, W[i], B, i)
  })
  // console.log(output)
  let outputSM = functions.softmax(output)
  // console.log(outputSM)
    return {output:outputSM, l2: [output,outputSM]}
}

}

function backprop(oData, Y) {
 
}

function gradient() {
  
}
