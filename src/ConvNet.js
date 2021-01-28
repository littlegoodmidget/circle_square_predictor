class ImageVerification {
    constructor(r, convNet, inputHeight) {
        this.convNet = convNet;
        this.inputHeight = inputHeight;
        this.r = r;

        this.learningRate = 0.01;

        this.inputs = [];
        this.output;
        this.preOutput;

        this.weights = [];
        this.bias = (Math.random() - 0.5) * 2 * r;
        this.weights = nnmath.createMatrix(1, this.inputHeight, this.r);
    }
    feedFowards(i1, i2) {
        if (this.inputHeight != i1.length || this.inputHeight != i2.length) throw new Error('Incorrect Sizing of Matrices!!');

        this.inputs[0] = i1;
        this.inputs[1] = i2;

        this.preOutput = [];
        this.preOutput[0] = [0]
        for(let i = 0; i < this.weights.length; i++) {
            this.preOutput[0][0] += this.weights[i] * (i1[0] - i2[0]) ** 2;
        }
        this.preOutput[0][0] += this.bias;

        this.output[0][0] = nnmath.sigmoid(this.preOutput[0][0]);

        return this.output;
    }
}


class ConvNet {
    static StoreAsJSON(convNet, fileName = 'model') {
        let data = {};
        let template = convNet.template;


        for(let i = 0; i < convNet.layers.length; i++) {
            const type = template[i].type;
            let layer = convNet.layers[i];

            switch (type) {
                case 'ConvLayer':
                    data[i.toString()] = {
                        type: 'ConvLayer',
                        inputShape: layer.inputShape,
                        filterShape: layer.filterShape,
                        numOfOutputs: layer.numOfOutputs,
                        settings: layer.settings,
                        filters: layer.filters,
                        biases: layer.biases,
                        learningRate: layer.learningRate
                    }
                    break;
                case 'MaxPool':
                    data[i.toString()] = {
                        type: 'MaxPool',
                        inputShape: layer.inputShape,
                        filterShape: layer.filterShape,
                        settings: layer.settings,
                    }
                    break;
                case 'FCAttachment':
                    data[i.toString()] = {
                        type: 'FCAttachment',
                        inputShape: layer.inputShape,
                    }
                    break;
                case 'FC':
                    data[i.toString()] = {
                        type: 'FC',
                        structure: layer.structure,
                        activations: layer.activations,
                        weights: layer.weights,
                        biases: layer.biases
                    }
                    break;
            }
        }

        const model = JSON.stringify(data);
        console.log(model, data);
        const blob = new Blob([model], {type: "application/json"})
        const modelObjectURL = URL.createObjectURL(blob);

        const anchorTag = document.createElement('a');
        anchorTag.href = modelObjectURL;
        anchorTag.download = fileName;
        anchorTag.click();
    }
    constructor(r, template, learningRate = 0.01) {
        this.r = r;
        this.learningRate = learningRate;
        this.template = template;

        this.output;

        
        this.batchSize = 32;
        // this.batchSize = 32;
        this.batchNum = 0;

        this.trainingLoss = [];
        this.trainingAccuracy = [];
        this.valLoss = [];
        this.valAccuracy = [];

        this.layers = [];
        for(let i = 0; i < template.length; i++) {
            const { inputShape, filterShape, numOfOutputs, settings, inputNodes, outputNodes, activation } = template[i];
            switch (template[i].type) {
                case 'ConvLayer':
                    this.layers[i] = new ConvLayer(
                        inputShape,
                        filterShape,
                        numOfOutputs,
                        settings,
                        this.learningRate,
                        r
                    );
                    break;
                case 'MaxPool':
                    this.layers[i] = new MaxPool(
                        inputShape,
                        filterShape,
                        settings
                    );
                    break;
                case 'FCAttachment':
                    this.layers[i] = new FCAttachment(
                        inputShape
                    );
                    break;
                case 'FC':
                    this.layers[i] = new Layer(r / 4, inputNodes, outputNodes, activation, learningRate);
                    break;
                default:
                    throw new Error('Unknown Layer Type');
            }
        }
    }
    feedFowards(input) {
        for(let i = 0; i < this.layers.length; i++) {
            input = this.layers[i].feedFowards(input);
        }

        this.output = input;

        return input;
    }
    backPropagation(target) {
        this.batchNum++;

        let updateGradients = false;
        if (this.batchNum >= this.batchSize) {
            this.batchNum = 0;
            updateGradients = true;
        }

        let gradient = [];
        for(let i = 0; i < target.length; i++) {
            gradient[i] = [this.output[i][0] - target[i][0]];
        }

        // console.log(gradient)

        // let gradient = this.layers[this.layers.length - 1].backPropagation(target, this.output, this.batchSize, updateGradients, this.learningRate); 
        for(let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backPropagation(gradient, updateGradients, this.batchSize);
        }
        
        // let loss = this.layers[this.layers.length - 1].error(target, this.output).totalError;
        target = null;
        // console.log(loss);

    }
    computeAverageEpochLoss(dataset) {
        let totalLoss = 0;
        let trainingAccuracy = 0;
        for(let i = 0; i < dataset.length; i++) {
            // let index = ~~(Math.random() * dataset.length);
            let data = dataset[i];

            let prediction = this.feedFowards(data.input);

            
            // totalLoss += this.layers[this.layers.length - 1].error(data.target, this.output).totalError;
            let loss = [];
            for(let i = 0; i < data.target.length; i++) {
                loss[i] = [1/ 2 * (data.target[i][0] - prediction[i][0]) ** 2];
                totalLoss += loss[i][0];
            }
            if (totalLoss < 0) {
                debugger
            }


            let maxIndex = -1;
            let maxValue = -Infinity;

            let targetMaxIndex = -1;
            let targetMaxValue = -Infinity;

            for(let i = 0; i < prediction.length; i++) {
                if (prediction[i][0] > maxValue) {
                    maxIndex = i;
                    maxValue = prediction[i][0];
                }
                if (data.target[i][0] > targetMaxValue) {
                    targetMaxIndex = i;
                    targetMaxValue = data.target[i][0];
                }
            }

            if (maxIndex === targetMaxIndex) {
                trainingAccuracy++;
            }
        }
        // this.loss.push(totalLoss / dataset.length);
        // this.trainingAccuracy.push(trainingAccuracy / dataset.length);
        // console.log(this.loss);
        return {averageLoss: totalLoss / dataset.length, averageAccuracy: trainingAccuracy / dataset.length};
    }
}

class ConvLayer {
    constructor(inputShape, filterShape, numOfOutputs, settings, learningRate = 0.001, r = 1) {
        this.inputShape = inputShape;
        this.filterShape = filterShape;
        this.numOfOutputs = numOfOutputs;
        this.settings = settings;

        this.learningRate = learningRate;
        this.r = r;


        this.momentum = 0.9;
        
        this.filterChanges = [];
        this.biasChanges = [];

        this.vFilterChanges = [];
        this.vBiasChanges = [];
        this.previousVFilterChanges = [];
        this.previousVBiasChanges = [];

        this.biases = [];
        this.filters = [];
        for(let i = 0; i < numOfOutputs; i++) {
            this.filters[i] = [];
            this.biases[i] = (Math.random() - 0.5) * 2 * r;

            this.filterChanges[i] = [];
            this.biasChanges[i] = 0;

            this.vFilterChanges[i] = [];
            this.vBiasChanges[i] = 0;
            this.previousVFilterChanges[i] = [];
            this.previousVBiasChanges[i] = 0;
            
            
            for(let j = 0; j < inputShape[2]; j++) {
                this.filters[i][j] = nnmath.createMatrix(filterShape[0], filterShape[1], r);
                this.filterChanges[i][j] = nnmath.createFilledMatrix(filterShape[0], filterShape[1], 0);

                this.vFilterChanges[i][j] = nnmath.createFilledMatrix(filterShape[0], filterShape[1], 0);
                this.previousVFilterChanges[i][j] = nnmath.createFilledMatrix(filterShape[0], filterShape[1], 0);
            }
        }


        this.extraZeros = (this.inputShape[0] - this.filterShape[0] + 2 * this.settings.padding) % this.settings.stride;

        this.input;
        this.preOutput;
        this.output;
    }
    feedFowards(input) {
        expectedInputShape(input, this.inputShape);

        this.input = input;

        // if (this.preOutput) {
        //     for(let i = 0; i < this.preOutput.length; i++) {
        //         for(let j = 0; j < this.preOutput[i].length; j++) {
        //             for(let z = 0; z < this.preOutput[i][j].length; z++) {
        //                 delete this.preOutput[i][j][z];
        //             }
        //             this.preOutput[i][j] = null;
        //         }
        //         this.preOutput[i] = null;
        //     }
        //     this.preOutput = null;
        // }
        
        this.output = null;
        this.output = [];
        this.preOutput = [];
        for(let i = 0; i < this.filters.length; i++) {
            this.preOutput[i] = conv2D(this.input, this.filters[i], this.settings);

            for(let j = 0; j < this.preOutput[0].length; j++) {
                for(let z = 0; z < this.preOutput[0][0].length; z++) {
                    this.preOutput[i][j][z] += this.biases[i];
                }
            }

            this.output[i] = nnmath.reluMat(this.preOutput[i]);

        }

        input = null;

        // can move this in loop above
        // for(let i = 0; i < this.preOutput.length; i++) {
        // }

        return this.output;
    }
    backPropagation(gradients, updateGradients, batchSize) {
        expectedInputShape(gradients, [this.output[0][0].length, this.output[0].length, this.output.length]);

        // this.learningRate = learningRate;

        let lossWRTCLBiases = [];
        let lossWRTCLPreOutput = [];
        for(let i = 0; i < gradients.length; i++) {
            lossWRTCLPreOutput[i] = nnmath.elementwiseMultiplication(nnmath.reluDeriMat(this.preOutput[i]), gradients[i]);

            let sumOfGradients = 0;
            for(let j = 0; j < lossWRTCLPreOutput[0].length; j++) {
                for(let z = 0; z < lossWRTCLPreOutput[0][0].length; z++) {
                    sumOfGradients += lossWRTCLPreOutput[i][j][z];
                }
            }

            lossWRTCLBiases[i] = sumOfGradients;

            // this.biases[i] -= this.learningRate * sumOfGradients;
            this.biasChanges[i] += sumOfGradients;
        }
        // this.biasChanges.push(lossWRTCLBiases);

        let lossWRTCLInput = [];
        for(let i = 0; i < this.input.length; i++) {
            let newInputs = [];
            let newFilters = [];
            for(let j = 0; j < lossWRTCLPreOutput.length; j++) {
                newInputs.push(nnmath.padGridItems(lossWRTCLPreOutput[j], this.settings.stride - 1, this.extraZeros));
                newFilters.push(nnmath.roatedMatrixVertHor(this.filters[j][i]));
            }
            lossWRTCLInput[i] = conv2D(newInputs, newFilters, {stride: 1, padding: this.filterShape[0] - 1 - this.settings.padding});
        }
        
        // console.log(lossWRTCLInput);
        expectedInputShape(lossWRTCLInput, this.inputShape);

        

        let filterGradients = [];
        for(let i = 0; i < lossWRTCLPreOutput.length; i++) {
            // For each kth new filter, apply convolution to kth newFilter for kth filter at nth depth
            let newFilter = nnmath.padGridItems(lossWRTCLPreOutput[i], this.settings.stride - 1, this.extraZeros);

            filterGradients[i] = [];
            for(let j = 0; j < this.input.length; j++) {
                filterGradients[i][j] = singleConv2D(this.input[j], newFilter, {stride: 1, padding: this.settings.padding});
            }

            for(let j = 0; j < filterGradients[0].length; j++) {
                for(let z = 0; z < filterGradients[0][0].length; z++) {
                    for(let k = 0; k < filterGradients[0][0][0].length; k++) {
                        // this.filters[i][j][z][k] -= this.learningRate * filterGradients[i][j][z][k];
                        this.filterChanges[i][j][z][k] += filterGradients[i][j][z][k];
                    }
                }
            }
        }
        // this.filterChanges.push(filterGradients)


        expectedInputShape(filterGradients[0], [this.filters[0][0][0].length, this.filters[0][0].length, this.filters[0].length]);
        expectedInputShape(filterGradients, [this.filters[0][0].length, this.filters[0].length, this.filters.length]);
        // console.log(filterGradients);
        gradients = null;
        filterGradients = null;
        lossWRTCLBiases = null;
        lossWRTCLPreOutput = null;

        if (updateGradients) {
            // console.log(updateGradients)
            for(let i = 0; i < this.filterChanges.length; i++) {
                for(let j = 0; j < this.filterChanges[i].length; j++) {
                    for(let z = 0; z < this.filterChanges[i][j].length; z++) {
                        for(let k = 0; k < this.filterChanges[i][j][z].length; k++) {
                            this.vFilterChanges[i][j][z][k] = this.momentum * this.previousVFilterChanges[i][j][z][k] + this.learningRate * this.filterChanges[i][j][z][k] / batchSize;
                            this.previousVFilterChanges[i][j][z][k] = this.vFilterChanges[i][j][z][k];
                            
                            this.filterChanges[i][j][z][k] = 0;
                            
                            // this.filters[i][j][z][k] -= this.learningRate * this.filterChanges[i][j][z][k] / batchSize;
                            this.filters[i][j][z][k] -= this.vFilterChanges[i][j][z][k];
                        }
                    }
                }
            }
            for(let i = 0; i < this.biasChanges.length; i++) {
                this.vBiasChanges[i] = this.momentum * this.previousVBiasChanges[i] + this.learningRate * this.biasChanges[i] / batchSize;

                this.previousVBiasChanges[i] = this.vBiasChanges[i];
                
                this.biasChanges[i] = 0;
                
                // this.biases[i] -= this.learningRate * this.biasChanges[i] / batchSize;
                this.biases[i] -= this.vBiasChanges[i];
            }
        }

        return lossWRTCLInput;
    }
}

// class ConvLayer {
//     constructor(inputShape, filterShape, numOfOutputs, settings, learningRate = 0.001, r = 1) {
//         this.inputShape = inputShape;
//         this.filterShape = filterShape;
//         this.numOfOutputs = numOfOutputs;
//         this.settings = settings;

//         this.learningRate = learningRate;
//         this.r = r;

//         this.filterChanges = [];
//         this.biasChanges = [];

//         this.biases = [];
//         this.filters = [];
//         for(let i = 0; i < numOfOutputs; i++) {
//             this.biases[i] = (Math.random() - 0.5) * 2 * r;
//             this.filters[i] = [];
//             this.biasChanges[i] = 0;
//             this.filterChanges[i] = [];
//             for(let j = 0; j < inputShape[2]; j++) {
//                 this.filters[i][j] = nnmath.createMatrix(filterShape[0], filterShape[1], r);
//                 this.filterChanges[i][j] = nnmath.createFilledMatrix(filterShape[0], filterShape[1], 0);
//             }
//         }


//         this.extraZeros = (this.inputShape[0] - this.filterShape[0] + 2 * this.settings.padding) % this.settings.stride;

//         this.input;
//         this.preOutput;
//         this.output;
//     }
//     feedFowards(input) {
//         expectedInputShape(input, this.inputShape);

//         this.input = input;

//         // if (this.preOutput) {

//         //     for(let i = 0; i < this.preOutput.length; i++) {
//         //         for(let j = 0; j < this.preOutput[i].length; j++) {
//         //             for(let z = 0; z < this.preOutput[i][j].length; z++) {
//         //                 delete this.preOutput[i][j][z];
//         //             }
//         //             this.preOutput[i][j] = null;
//         //         }
//         //         this.preOutput[i] = null;
//         //     }
//         //     this.preOutput = null;
//         // }
        
//         this.output = null;
//         this.output = [];
//         this.preOutput = [];
//         for(let i = 0; i < this.filters.length; i++) {
//             this.preOutput[i] = conv2D(this.input, this.filters[i], this.settings);

//             for(let j = 0; j < this.preOutput[0].length; j++) {
//                 for(let z = 0; z < this.preOutput[0][0].length; z++) {
//                     this.preOutput[i][j][z] += this.biases[i];
//                 }
//             }

//             this.output[i] = nnmath.reluMat(this.preOutput[i]);

//         }

//         input = null;

//         // can move this in loop above
//         // for(let i = 0; i < this.preOutput.length; i++) {
//         // }

//         return this.output;
//     }
//     backPropagation(gradients, updateGradients, batchSize) {
//         expectedInputShape(gradients, [this.output[0][0].length, this.output[0].length, this.output.length]);

//         // this.learningRate = learningRate;

//         let lossWRTCLBiases = [];
//         let lossWRTCLPreOutput = [];
//         for(let i = 0; i < gradients.length; i++) {
//             lossWRTCLPreOutput[i] = nnmath.elementwiseMultiplication(nnmath.reluDeriMat(this.preOutput[i]), gradients[i]);

//             let sumOfGradients = 0;
//             for(let j = 0; j < lossWRTCLPreOutput[0].length; j++) {
//                 for(let z = 0; z < lossWRTCLPreOutput[0][0].length; z++) {
//                     sumOfGradients += lossWRTCLPreOutput[i][j][z];
//                 }
//             }

//             lossWRTCLBiases[i] = sumOfGradients;

//             // this.biases[i] -= this.learningRate * sumOfGradients;
//             this.biasChanges[i] += sumOfGradients;
//         }
//         // this.biasChanges.push(lossWRTCLBiases);

//         let lossWRTCLInput = [];
//         for(let i = 0; i < this.input.length; i++) {
//             let newInputs = [];
//             let newFilters = [];
//             for(let j = 0; j < lossWRTCLPreOutput.length; j++) {
//                 newInputs.push(nnmath.padGridItems(lossWRTCLPreOutput[j], this.settings.stride - 1, this.extraZeros));
//                 newFilters.push(nnmath.roatedMatrixVertHor(this.filters[j][i]));
//             }
//             lossWRTCLInput[i] = conv2D(newInputs, newFilters, {stride: 1, padding: this.filterShape[0] - 1 - this.settings.padding});
//         }
        
//         // console.log(lossWRTCLInput);
//         expectedInputShape(lossWRTCLInput, this.inputShape);

        

//         let filterGradients = [];
//         for(let i = 0; i < lossWRTCLPreOutput.length; i++) {
//             // For each kth new filter, apply convolution to kth newFilter for kth filter at nth depth
//             let newFilter = nnmath.padGridItems(lossWRTCLPreOutput[i], this.settings.stride - 1, this.extraZeros);

//             filterGradients[i] = [];
//             for(let j = 0; j < this.input.length; j++) {
//                 filterGradients[i][j] = singleConv2D(this.input[j], newFilter, {stride: 1, padding: this.settings.padding});
//             }

//             for(let j = 0; j < filterGradients[0].length; j++) {
//                 for(let z = 0; z < filterGradients[0][0].length; z++) {
//                     for(let k = 0; k < filterGradients[0][0][0].length; k++) {
//                         // this.filters[i][j][z][k] -= this.learningRate * filterGradients[i][j][z][k];
//                         this.filterChanges[i][j][z][k] += filterGradients[i][j][z][k];
//                     }
//                 }
//             }
//         }
//         // this.filterChanges.push(filterGradients)


//         expectedInputShape(filterGradients[0], [this.filters[0][0][0].length, this.filters[0][0].length, this.filters[0].length]);
//         expectedInputShape(filterGradients, [this.filters[0][0].length, this.filters[0].length, this.filters.length]);
//         // console.log(filterGradients);
//         gradients = null;
//         filterGradients = null;
//         lossWRTCLBiases = null;
//         lossWRTCLPreOutput = null;

//         if (updateGradients) {
//             // console.log(updateGradients)
//             for(let i = 0; i < this.filterChanges.length; i++) {
//                 for(let j = 0; j < this.filterChanges[i].length; j++) {
//                     for(let z = 0; z < this.filterChanges[i][j].length; z++) {
//                         for(let k = 0; k < this.filterChanges[i][j][z].length; k++) {
//                             this.filters[i][j][k][z] -= this.learningRate * this.filterChanges[i][j][z][k] / batchSize;
//                             this.filterChanges[i][j][k][z] = 0;
//                         }
//                     }
//                 }
//             }
//             for(let i = 0; i < this.biasChanges; i++) {
//                 this.biases[i] -= this.learningRate * this.biasChanges[i] / batchSize;
//             }
//         }

//         return lossWRTCLInput;
//     }
// }

class MaxPool {
    constructor(inputShape, filterShape, settings) {
        this.inputShape = inputShape;
        this.filterShape = filterShape;
        this.settings = settings;

        this.input;
        this.output;
    }
    feedFowards(input) {
        expectedInputShape(input, this.inputShape);
        this.input = input;
        
        const {stride, padding} = this.settings;

        const newWidth = ~~((input[0][0].length - this.filterShape[0] + padding * 2) / stride) + 1;
        const newHeight = ~~((input[0].length - this.filterShape[1] + padding * 2) / stride) + 1;
        // console.log(newWidth, newHeight)
        this.output = null;
        this.output = [];
        for(let z = 0; z < input.length; z++) {
            this.output[z] = [];
            for(let i = 0; i < newHeight; i++) {
                this.output[z][i] = [];
                for(let j = 0; j < newWidth; j++) {

                    let max = -Infinity;
                    for(let fy = 0; fy < this.filterShape[1]; fy++) {
                        for(let fx = 0; fx < this.filterShape[0]; fx++) {
                            const y = i * stride + fy - padding;
                            const x = j * stride + fx - padding;

                            if (y < 0 || x < 0 || y >= this.inputShape[1] || x >= this.inputShape[0]) {
                                max = Math.max(0, max);
                                continue;
                            }

                            max = Math.max(input[z][y][x], max);
                        }
                    }

                    this.output[z][i][j] = max;
                    
                }
            }
        }

        return this.output;
    }
    backPropagation(gradients) {
        expectedInputShape(gradients, [this.output[0][0].length, this.output[0].length, this.output.length]);

        let lossWRTMPOutput = [];
        for(let i = 0; i < gradients.length; i++) {
            lossWRTMPOutput[i] = nnmath.createFilledMatrix(this.inputShape[0], this.inputShape[1], 0);
        }
        for(let i = 0; i < gradients.length; i++) {
            for(let j = 0; j < gradients[0].length; j++) {
                for(let z = 0; z < gradients[0][0].length; z++) {
                    for(let fy = 0; fy < this.filterShape[1]; fy++) {
                        for(let fx = 0; fx < this.filterShape[0]; fx++) {
                            const y = j * this.settings.stride + fy - this.settings.padding;
                            const x = z * this.settings.stride + fx - this.settings.padding;

                            if (y < 0 || x < 0 || y >= this.inputShape[1] || x >= this.inputShape[0]) continue;
                            
                            if (this.output[i][j][z] === this.input[i][y][x]) lossWRTMPOutput[i][y][x] = gradients[i][j][z];
                            else continue; //As we already filled it with zeros

                        }
                    }

                }
            }
        }

        gradients = null;

        expectedInputShape(lossWRTMPOutput, this.inputShape);

        return lossWRTMPOutput;
    }
}

class FCAttachment {
    constructor(inputShape) {
        this.inputShape = inputShape;

        this.input;
        this.output = [];
        for(let i = 0; i < this.inputShape[2]; i++) {
            this.output = [];
        }
    }
    feedFowards(input) {
        expectedInputShape(input, this.inputShape);

        this.input = input;
        
        let counter = 0;

        this.output = null;
        this.output = [];
        for(let i = 0; i < input.length; i++) {
            // this.output[i] = [];
            for(let j = 0; j < input[0].length; j++) {
                for(let z = 0; z < input[0][0].length; z++) {
                    this.output[counter] = [input[i][j][z]];
                    
                    counter++;
                }
            }
        }

        return this.output;
    }
    backPropagation(gradient) {
        let counter = 0;

        let transformedGradient = [];
        for(let i = 0; i < this.inputShape[2]; i++) {
            transformedGradient[i] = []
            for(let j = 0; j < this.inputShape[1]; j++) {
                transformedGradient[i][j] = []
                for(let z = 0; z < this.inputShape[0]; z++) {
                    transformedGradient[i][j][z] = gradient[counter][0];
                    counter++;
                }
            }
        }

        expectedInputShape(transformedGradient, this.inputShape);

        return transformedGradient;
    }
}

function expectedInputShape(input, inputShape) {
    if (input.length != inputShape[2] || input[0].length != inputShape[1] || input[0][0].length != inputShape[0]) throw new Error('Expected Input of: ' + [inputShape[0], inputShape[1], inputShape[2]] + ' but had gotten ' + [input[0][0].length, input[0].length, input.length]);
}