const canvas = document.createElement('canvas');
canvas.width = 48;
canvas.height = 48;
document.body.appendChild(canvas);
const c = canvas.getContext('2d');

const prediction = document.getElementById('prediction');

function saveCanvas() {
    let imageData = c.getImageData(0, 0, canvas.width, canvas.height);

    let alpha = [];
    for(let i = 0; i < imageData.data.length; i+=4) {
        const RED = imageData.data[i + 3];
        alpha.push(RED);
    }

    let data = [];
    for(let i = 0; i < imageData.height; i++) {
        data[i] = [];
        for(let j = 0; j < imageData.width; j++) {
            data[i][j] = alpha[i * imageData.width + j] / 255;
        }
    }
    
    return [data];
}


const inputs = {
    Digit1: false,
    Digit2: false,
    Digit3: false,
    Digit4: false,
    T: false,
    keydown(e) {
        let down = e.type === 'keydown';

        switch (e.code) {
            case 'Digit1':
                this.Digit1 = down;
                break;
            case 'Digit2':
                this.Digit2 = down;
                break;
            case 'Digit3':
                this.Digit3 = down;
                break;
            case 'Digit4':
                this.Digit4 = down;
                break;
            case 'KeyT':
                this.T = down;
                break;
            case 'Enter':
                if (down) {
                    let canvasPixels = saveCanvas();
                    shapeNet.feedFowards(canvasPixels);
                    
                    prediction.innerText = shapeNet.output[0][0] > shapeNet.output[1][0]? 'Square': 'Circle';

                    c.clearRect(0, 0, canvas.width, canvas.height);
                }
                break;
        }
    }
};



window.addEventListener('keydown', function(e) {
    if (!e.repeat) inputs.keydown(e);
})
window.addEventListener('keyup', function(e) {
    inputs.keydown(e);
})

let mouse = {
    x: 0,
    y: 0,
    down: false
}

let mouseMoveCounter = 0;
canvas.addEventListener('mousemove', function(e) {
    mouseMoveCounter++;
    mouse.x = e.offsetX * (canvas.width / 600);
    mouse.y = e.offsetY * (canvas.height / 600);
    if (mouse.down) {
        c.beginPath();
        c.arc(mouse.x, mouse.y, 1, 0, Math.PI * 2);
        c.fill();

        if (mouseMoveCounter % 10 === 0) {
            console.log('prediction')
            let canvasPixels = saveCanvas();
            shapeNet.feedFowards(canvasPixels);
            
            prediction.innerText = shapeNet.output[0][0] > shapeNet.output[1][0]? 'Square': 'Circle';
        }
    }
})

canvas.addEventListener('mousedown', function(e) {
    mouse.down = true;
})
canvas.addEventListener('mouseup', function(e) {
    mouse.down = false;
})