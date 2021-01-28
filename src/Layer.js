class NeuralNetwork {
    constructor(structure, activations, r, batchSize) {
        this.structure = structure;
        this.activations = activations;
        this.r = r;

        this.batchSize = batchSize;
        this.batchNum = 0;

        this.output;

        this.layers = [];
        for(let i = 0; i < this.structure.length - 1; i++) {
            let w = this.structure[i];
            let h = this.structure[i + 1];
            this.layers[i] = new Layer(this.r, w, h, this.activations[i]);
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

        // Loss = 1/2(Target - Output) ** 2;

        let gradient = [];
        for(let i = 0; i < target.length; i++) {
            gradient[i] = [this.output[i][0] - target[i][0]];
        }

        this.batchNum++;
        let computeGradients = false;
        if (this.batchNum >= this.batchSize) {
            computeGradients = true;
            this.batchNum = 0;
        }

        // Couple speed it up by making a completely seperate function that runs only if computeGradients is true
        
        for(let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backPropagation(gradient, computeGradients, this.batchSize);
        }

        return gradient;
    }
}


class Layer {
    constructor(r, inputNodes, outputNodes, activation, learningRate) {
        this.inputNodes = inputNodes;
        this.outputNodes = outputNodes;
        this.r = r;
        this.activation = nnmath[activation + 'Mat'].bind(nnmath);
        this.activationDeri = nnmath[activation + 'DeriMat'].bind(nnmath);

        this.learningRate = learningRate;
        
        this.input;
        this.net;
        this.output;

        this.momentum = 0.90;

        this.weights = nnmath.createMatrix(this.inputNodes, this.outputNodes, this.r);
        this.biases = nnmath.createMatrix(1, this.outputNodes, this.r);

        this.weightAcc = nnmath.createFilledMatrix(this.inputNodes, this.outputNodes, 0);
        this.biasesAcc = nnmath.createFilledMatrix(1, this.outputNodes, 0);
        // this.previousWeightAcc = nnmath.createFilledMatrix(this.inputNodes, this.outputNodes, 0);
        // this.previousBiasesAcc = nnmath.createFilledMatrix(1, this.outputNodes, 0);
        this.vWeightAcc = nnmath.createFilledMatrix(this.inputNodes, this.outputNodes, 0);
        this.vBiasesAcc = nnmath.createFilledMatrix(1, this.outputNodes, 0);
        this.previousVWeightAcc = nnmath.createFilledMatrix(this.inputNodes, this.outputNodes, 0);
        this.previousVBiasesAcc = nnmath.createFilledMatrix(1, this.outputNodes, 0);
    }
    feedFowards(input) {
        this.input = input;

        this.net = nnmath.add(
            nnmath.dotProduct(
                this.weights,
                this.input
            ), 
            this.biases
        );
        this.output = this.activation(
            this.net
        );


        return this.output;
    }
    backPropagation(gradient, computeGradients, batchSize) {
        const lossWRTNet = nnmath.elementwiseMultiplication(this.activationDeri(this.net), gradient);


        if (!computeGradients) {
            for(let i = 0; i < this.weights.length; i++) {
                for(let j = 0; j < this.input.length; j++) {    
                    this.weightAcc[i][j] += this.input[j][0] * lossWRTNet[i][0];
                }
                this.biasesAcc[i][0] += lossWRTNet[i][0];
            }
        }   else {
            for(let i = 0; i < this.weights.length; i++) {
                for(let j = 0; j < this.input.length; j++) {    
                    this.weightAcc[i][j] += this.input[j][0] * lossWRTNet[i][0];

                    
                    this.vWeightAcc[i][j] = this.momentum * this.previousVWeightAcc[i][j] + 0.2 * this.weightAcc[i][j] / batchSize;
                    this.weightAcc[i][j] = 0;
                    
                    this.previousVWeightAcc[i][j] = this.vWeightAcc[i][j];

                    this.weights[i][j] -= this.vWeightAcc[i][j];
                }
                this.biasesAcc[i][0] += lossWRTNet[i][0];

                this.vBiasesAcc[i][0] = this.momentum * this.previousVBiasesAcc[i][0] + 0.2 * this.biasesAcc[i][0] / batchSize;
                this.biasesAcc[i][0] = 0;

                this.previousVBiasesAcc[i][0] = this.vBiasesAcc[i][0];

                this.biases[i][0] -= this.vBiasesAcc[i][0];
            }
        }

        const lossWRTInput = nnmath.dotProduct(nnmath.transpose(this.weights), lossWRTNet);
        return lossWRTInput;
    }
}