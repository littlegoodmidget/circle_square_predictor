function conv2D(input, filter, settings) {

    if (input.length != filter.length) throw new Error('Incorrect Matrix Sizing!');

    const {stride, padding} = settings;

    const newWidth = ~~((input[0][0].length - filter[0][0].length + padding * 2) / stride) + 1;
    const newHeight = ~~((input[0].length - filter[0].length + padding * 2) / stride) + 1;


    let sum = 0;
    let output = [];
    for(let i = 0; i < newHeight; i++) {
        output[i] = [];
        for(let j = 0; j < newWidth; j++) {

            sum = 0;

            for(let fy = 0; fy < filter[0].length; fy++) {
                for(let fx = 0; fx < filter[0][0].length; fx++) {
                    const y = i * stride + fy - padding;
                    const x = j * stride + fx - padding;
                    if (y < 0 || x < 0 || y >= input[0].length || x >= input[0][0].length) continue;
                    for(let z = 0; z < input.length; z++) {

                        sum += input[z][y][x] * filter[z][fy][fx];

                    }   
                }
            }
            output[i][j] = sum;
            
        }


    }
    

    return output;
}

function singleConv2D(input, filter, settings) {
    
    // if (input.length != filter.length) throw new Error('Incorrect Matrix Sizing!');

    const {stride, padding} = settings;

    const newWidth = ~~((input[0].length - filter[0].length + padding * 2) / stride) + 1;
    const newHeight = ~~((input.length - filter.length + padding * 2) / stride) + 1;


    let sum = 0;
    let output = [];
    for(let i = 0; i < newHeight; i++) {
        output[i] = [];
        for(let j = 0; j < newWidth; j++) {

            sum = 0;

            for(let fy = 0; fy < filter.length; fy++) {
                for(let fx = 0; fx < filter[0].length; fx++) {
                    const y = i * stride + fy - padding;
                    const x = j * stride + fx - padding;
                    if (y < 0 || x < 0 || y >= input.length || x >= input[0].length) continue;
                    sum += input[y][x] * filter[fy][fx];
                }
            }
            output[i][j] = sum;
            
        }


    }
    

    return output;
}