class NN {
    static error(target, output) {
        if (target.length != output.length) throw new Error("Target Does Not Match Output Dimensions!");
        let data = {
            error: [],
            totalError: 0
        }
        for(let i = 0; i < target.length; i++) {
            data.error[i] = [0.5 * (target[i][0] - output[i][0]) ** 2];
            data.totalError += data.error[i][0];
        }
        return data;
    }
    static errorDeriv(target, output) {
        if (target.length != output.length) throw new Error("Target Does Not Match Output Dimensions!");
        let data = {
            gradient: [],
        }
        for(let i = 0; i < target.length; i++) {
            data.gradient[i] = [output[i][0] - target[i][0]];
        }
        return data;
    }
    constructor(r, structure, activations, learningRate = 0.01) {
        this.r = r;
        this.structure = structure;
        this.activations = activations;
        this.learningRate = learningRate;

        this.activationFunctions = [];
        this.activationFunctionDerivatives = [];

        if (this.structure.length - 1 != this.activations.length) throw new Error("Activations Inputted and Layers Inputted do not match");

        this.layerInputs = [];
        this.layerOutputs = [];

        this.weights = [];
        this.biases = [];

        this.weightChanges = [];
        this.biasChanges = [];

        for(let i = 0; i < this.structure.length - 1; i++) {
            let width = this.structure[i];
            let height = this.structure[i + 1];

            if (activations[i] != 'softmax') {
                this.activationFunctions[i] = (nnmath[activations[i] + "Mat"].bind(nnmath));
                this.activationFunctionDerivatives[i] = (nnmath[activations[i] + "DeriMat"].bind(nnmath));
            }   else {
                this.activationFunctions[i] = this.softmax;
                this.activationFunctionDerivatives[i] = this.softmaxDeri;

                this.softMaxSum = 0;
            }
            this.weights[i] = nnmath.createMatrix(width, height, this.r);
            this.biases[i] = nnmath.createMatrix(1, height, this.r);

            this.weightChanges[i] = nnmath.createFilledMatrix(width, height, 0);
            this.biasChanges[i] = nnmath.createFilledMatrix(1, height, 0);
        }
    }
    error(target, output) {
        if (target.length != output.length) throw new Error("Target Does Not Match Output Dimensions!");
        let data = {
            error: [],
            totalError: 0
        }
        for(let i = 0; i < target.length; i++) {
            data.error[i] = [0.5 * (target[i][0] - output[i][0]) ** 2];
            data.totalError += data.error[i][0];
        }
        return data;
    }
    errorDeriv(target, output) {
        if (target.length != output.length) throw new Error("Target Does Not Match Output Dimensions!");
        let data = {
            gradient: [],
        }
        for(let i = 0; i < target.length; i++) {
            data.gradient[i] = [output[i][0] - target[i][0]];
        }
        return data;
    }
    softmax(input) {
        const t = [];
        let sum = 0;
        for(let i = 0; i < input.length; i++) {
            t[i] = [Math.exp(input[i])];
            sum += t[i][0];
        }

        this.softmaxSum = sum;
        
        const o = [];
        for(let i = 0; i < input.length; i++) {
            o[i] = [t[i][0] / sum];
        }

        return o;
    }
    softmaxDeri(t) {
        // console.log(this.);

        return nnmath.createFilledMatrix(t[0].length, t.length, 1);
    }
    feedFowards(inputs) {
        let output = inputs;
        this.layerInputs[0] = inputs;
        this.layerOutputs[0] = inputs;
        
        for(let i = 0; i < this.weights.length; i++) {
            const NET = nnmath.add(nnmath.dotProduct(this.weights[i], output), this.biases[i]);
            this.layerInputs[i + 1] = NET;
            output = this.activationFunctions[i](NET);
            this.layerOutputs[i + 1] = output;
        }
        return output;
    }
    backPropagation(target, output, batchSize, updateGradients, learningRate) {
        let gradient = this.computeGradients(target, output);
        let gradientInfo = this.gradientInfo;

        this.learningRate = learningRate;
        
        for(let i = 0; i < this.weights.length; i++) {
            for(let j = 0; j < this.weights[i].length; j++) {
                for(let z = 0; z < this.weights[i][j].length; z++) {
                    // this.weights[i][j][z] -= this.learningRate * gradientInfo.weights[i][j][z];
                    this.weightChanges[i][j][z] += gradientInfo.weights[i][j][z];
                }
                // this.biases[i][j][0] -= this.learningRate * gradientInfo.biases[i][j][0];
                this.biasChanges[i][j][0] += this.gradientInfo.biases[i][j][0];
            }
        }

        if (updateGradients) {
            for(let i = 0; i < this.weights.length; i++) {
                for(let j = 0; j < this.weights[i].length; j++) {
                    for(let z = 0; z < this.weights[i][j].length; z++) {
                        this.weights[i][j][z] -= this.learningRate * this.weightChanges[i][j][z] / batchSize;
                        this.weightChanges[i][j][z] = 0;
                        // this.weightChanges[i][j][z] += gradientInfo.weights[i][j][z];
                    }
                    this.biases[i][j][0] -= this.learningRate * this.biasChanges[i][j][0] / batchSize; 
                    this.biasChanges[i][j][0] = 0;
                    // this.biasChanges[i][j][0] += this.gradientInfo.biases[i][j][0];
                }
            }
        }

        return gradient;
    }
    // Will need to pass input later on somehow
    computeGradients(target, output) {
        // const errorInfo = this.error(target, output);
        // console.log(errorInfo.totalError);
        let weightChanges = [];
        let biasChanges = [];

        // console.log(errorInfo);

        const errorGradientWRTOutput = this.errorDeriv(target, output).gradient;



        let currentError = errorGradientWRTOutput;
        for(let i = this.layerOutputs.length - 1; i >= 1; i--) {
            const unextendedErrorGradientWRTNet = nnmath.elementwiseMultiplication(this.activationFunctionDerivatives[i - 1](this.layerInputs[i]), currentError);
            // const unextendedErrorGradientWRTNet = nnmath.elementwiseMultiplication(nnmath.sigmoidDeriMat(this.layerInputs[i]), currentError);
            const errorGradientWRTNET = nnmath.extendVectorToMat(unextendedErrorGradientWRTNet, this.layerOutputs[i - 1].length);
            // debugger
            if (this.weights[i - 1].length != errorGradientWRTNET.length) throw new Error("Incorrect Sizing of Matrices!");
            const NETGradientWRTWeights = nnmath.transpose(nnmath.extendVectorToMat(this.layerOutputs[i - 1], this.weights[i - 1].length));



            const errorGradientWRTWeights = nnmath.elementwiseMultiplication(
                errorGradientWRTNET,
                NETGradientWRTWeights
            );


            weightChanges[i - 1] = errorGradientWRTWeights;
            biasChanges[i - 1] = unextendedErrorGradientWRTNet;
            
            // if (i >= 2) {
            //     currentError = nnmath.dotProduct(nnmath.transpose(this.weights[i - 1]), unextendedErrorGradientWRTNet);
            // }
            currentError = nnmath.dotProduct(nnmath.transpose(this.weights[i - 1]), unextendedErrorGradientWRTNet);
        }

        this.gradientInfo = {weights: weightChanges, biases: biasChanges};
        // console.log(currentError);
        return currentError;
    }
    feedFowardsAndUpdate(target, input) {
        const output = this.feedFowards(input);
        const errorInfo = this.error(target, output);
        console.log(errorInfo);
        const gradientInfo = this.computeGradients(target, output);

        for(let i = 0; i < this.weights.length; i++) {
            for(let j = 0; j < this.weights[i].length; j++) {
                for(let z = 0; z < this.weights[i][j].length; z++) {
                    this.weights[i][j][z] -= this.learningRate * gradientInfo.weights[i][j][z];
                }
                this.biases[i][j][0] -= this.learningRate * gradientInfo.biases[i][j][0];
            }
        }
    }
}