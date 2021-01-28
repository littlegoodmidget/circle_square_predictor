const graph = document.getElementById('graph');
graph.width = 600;
graph.height = 600;
const c2 = graph.getContext('2d');


function graphItems(max_y_value, y_values, x_values = []) {
    console.log('draw')
    c2.clearRect(0, 0, graph.width, graph.height);
    c2.fillStyle = 'white';
    c2.fillRect(0, 0, graph.width, graph.height);
    c2.fillStyle = 'black';


    c2.font = '10px Arial';
    c2.textAlign = 'center'

    c2.beginPath();
    c2.moveTo(0, graph.height);
    for(let j = 0; j < y_values.length; j++) {
        c2.lineTo(graph.width / (y_values.length - 1) * j, graph.height - graph.height * y_values[j] / max_y_value);

        if (x_values.length === y_values.length) {
            c2.fillText(x_values[j], graph.width / (y_values.length - 1) * j, graph.height);
        }   else {
            c2.fillText(j, graph.width / (y_values.length - 1) * j, graph.height);
        }
    }
    c2.moveTo(graph.width, graph.height);

    c2.stroke();
}




let testInput = [];
for(let i = 0; i < 1; i++) {
    testInput[i] = [];
    for(let j = 0; j < 4; j++) {
        testInput[i][j] = [];
        for(let z = 0; z < 4; z++) {
            testInput[i][j][z] = Math.random();
            
        }
    }
}

let cn = new ConvNet(1, [
    {
        type: 'ConvLayer',
        inputShape: [4, 4, 1],
        filterShape: [3, 3],
        numOfOutputs: 10,
        settings: {
            stride: 1,
            padding: 0
        }
    },
    {
        type: 'MaxPool',
        inputShape: [2, 2, 10],
        filterShape: [2, 2],
        settings: {
            stride: 2,
            padding: 0
        }
    },
    {
        type: 'FCAttachment',
        inputShape: [1, 1, 10]
    },
    {
        type: 'FC',
        inputNodes: 10,
        outputNodes: 8,
        activation: 'sigmoid'
    },
    {
        type: 'FC',
        inputNodes: 8,
        outputNodes: 1,
        activation: 'sigmoid'
    }
], 1)