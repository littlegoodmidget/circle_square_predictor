const nnmath = {
    padGridItems(m, gridPaddingSize, extraZeros) {
        let numOfPaddinY = m.length - 1;
        let numOfPaddinX = m[0].length - 1;
        let result;
        result = this.createFilledMatrix(m[0].length + numOfPaddinX * gridPaddingSize + extraZeros, m.length + numOfPaddinY * gridPaddingSize + extraZeros, 0);
        for(let i = 0; i < m.length; i++) {
            for(let j = 0; j < m[0].length; j++) {
                result[i * (gridPaddingSize + 1)][j * (gridPaddingSize + 1)] = m[i][j];
            }
        }
        m = null;
        return result;
    },
    roatedMatrixVertHor(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = m[m.length - 1 - i][m[0].length - 1 - j];
            }
        }
        m = null;
        return result;
    },
    createMatrix(w, h, r = 1) {
        let result = [];
        for(let i = 0; i < h; i++) {
            result[i] = [];
            for(let j = 0; j < w; j++) {
                result[i][j] = (Math.random() - 0.5) * 2 * r;
            }
        }
        return result;
    },
    createFilledMatrix(w, h, n) {
        let result = [];
        for(let i = 0; i < h; i++) {
            result[i] = [];
            for(let j = 0; j < w; j++) {
                result[i][j] = n;
            }
        }
        return result;
    },
    dotProduct(m1, m2) {
        // h = m1.length, w = m2[0].length
        if (m1[0].length != m2.length) {
            console.log(m1, m2);
            throw new Error("Incorrect Sizing of Matrices");
        }

        let result = [];
        for(let i = 0; i < m1.length; i++) {
            result[i] = [];
            for(let j = 0; j < m2[0].length; j++) {
                let sum = 0;
                for(let z = 0; z < m1[0].length; z++) {
                    sum += m1[i][z] * m2[z][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    },
    add(m1, m2) {
        let result = [];
        for(let i = 0; i < m1.length; i++) {
            result[i] = [];
            for(let j = 0; j < m1[0].length; j++) {
                result[i][j] = m1[i][j] + m2[i][j];
            }
        }
        return result;
    },
    subtract(m1, m2) {
        let result = [];
        for(let i = 0; i < m1.length; i++) {
            result[i][j] = [];
            for(let j = 0; j < m1[0].length; j++) {
                result[i][j] = m1[i][j] - m2[i][j];
            }
        }
        return result;
    },

    extendVectorToMat(m, width) {
        if (m[0].length != 1) throw new Error("Incorrect Sizing of Matrix!");
        if (width < 1) throw new Error("Incorrect Extension Size!");
        // else if (width === 1) console.warn('Extension of Vector to Size of 1');

        let result = []; 
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < width; j++) {
                result[i][j] = m[i][0];
            }
        }
        return result;
    },
    elementwiseMultiplication(m1, m2) {
        if (m1.length != m2.length || m1[0].length != m2[0].length) throw new Error("Matrices are of Incorrect Sizes");

        let result = [];
        for(let i = 0; i < m1.length; i++) {
            result[i] = [];
            for(let j = 0; j < m1[0].length; j++) {
                result[i][j] = m1[i][j] * m2[i][j];
            }
        }
        m1 = null;
        m2 = null;
        return result;
    },
    transpose(m) {
        let result = [];
        for(let i = 0; i < m[0].length; i++) {
            result[i] = [];
            for(let j = 0; j < m.length; j++) {
                result[i][j] = m[j][i];
            }
        }
        return result;
    },
    sumColumnsIntoVector(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [0];
            for(let j = 0; j < m[0].length; j++) {
                result[i][0] += m[i][j];
            }
        }
        return result;
    },
    
    sigmoid(n) {
        return 1 / (1 + Math.exp(-n))
    },
    sigmoidMat(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = this.sigmoid(m[i][j]);
            }
        }
        return result;
    },
    sigmoidDeri(n) {
        let s = this.sigmoid(n);
        return s * (1 - s);
    },
    sigmoidDeriMat(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = this.sigmoidDeri(m[i][j]);
            }
        }
        return result;
    },
    
    relu(n) {
        return n>=0? n: 0;
    },
    reluMat(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = this.relu(m[i][j]);
            }
        }
        return result;
    },
    reluDeri(n) {
        return n>=0? 1: 0;
    },
    reluDeriMat(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = this.reluDeri(m[i][j]);
            }
        }
        return result;
    },
    leakyRelu(n) {
        return n>=0? n: 0.01 * n;
    },
    leakyReluMat(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = this.leakyRelu(m[i][j]);
            }
        }
        return result;
    },
    leakyReluDeri(n) {
        return n>=0? 1: 0.01;
    },
    leakyReluDeriMat(m) {
        let result = [];
        for(let i = 0; i < m.length; i++) {
            result[i] = [];
            for(let j = 0; j < m[0].length; j++) {
                result[i][j] = this.leakyReluDeri(m[i][j]);
            }
        }
        return result;
    },

    // softmax(input) {
    //     const t = [];
    //     for(let i = 0; i < input.length; i++) {
    //         t[i] = Math.exp(input[i]);
    //     }
    // },
    // softmaxMat(input) {
    //     const t = [];
    //     let sum = 0;
    //     for(let i = 0; i < input.length; i++) {
    //         t[i] = [Math.exp(input[i])];
    //         sum += t[i][0];
    //     }

    //     const o = [];
    //     for(let i = 0; i < input.length; i++) {
    //         o[i] = [t[i][0] / sum];
    //     }

    //     return o;
    // },
    // softmaxDeri(input) {
    //     const t = [];
    //     for(let i = 0; i < input.length; i++) {
    //         t[i] = Math.exp(input[i]);
    //     }
    // },
    // softmaxDeriMat(input) {
    //     const t = [];
    //     let sum = 0;
    //     for(let i = 0; i < input.length; i++) {
    //         t[i] = [Math.exp(input[i])];
    //         sum += t[i][0];
    //     }

    //     const o = [];
    //     for(let i = 0; i < input.length; i++) {
    //         o[i] = [t[i][0] / sum];
    //     }

    //     return o;
    // },
}