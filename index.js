window.addEventListener('load', async function() { 

//Label 1
const txtLabel1 = document.querySelector('#txtLabel1'),

//Predictions
btnPredict = document.querySelector('#btnPredict'),
lPrediction = document.querySelector('#lPrediction'),

//Status for feed
statusModel = document.querySelector('#statusModel'),
statusVideo = document.querySelector('#statusVideo'),

checkModel = document.querySelector('#CheckModel'),
loadLabels = document.querySelector('#loadLabels'),
loadModel = document.querySelector('#loadModel'),

jsonUpload = document.getElementById('upload-json'),
weightsUpload = document.getElementById('upload-weights');





var flagPredicting = false; 

var isPredicting = false; 



const classifier = new imageClassifier(6);
classifier.initalise().then(function () {

    statusModel.textContent = 'Model Loaded';

});

classifier.createWebcam().then(function () {
    statusVideo.textContent = 'Webcam Initialised';
});



checkModel.addEventListener('click', function() {
    classifier.checkModel(); 
});



loadLabels.addEventListener('click', function() {
    classifier.addLabel('C');
    classifier.addLabel('F');
    classifier.addLabel('G');
    classifier.addLabel('N');
    classifier.addLabel('H');
    classifier.addLabel('Z'); 
});

loadModel.addEventListener('click', function() {
    classifier.loadModel(); 
});







loadModel.addEventListener('click', async function() {
    const model = await tf.loadLayersModel(
        tf.io.browserFiles( [jsonUpload.files[0], weightsUpload.files[0]]));

        classifier.loadModel(model);
}) 
// addLabel7.addEventListener('click', function() {
//     if(num7 == 0) {
//         txtLabel7.disabled = true; 
//     }
//     let label = txtLabel7.value.trim();  
//     classifier.addLabel(label);
//     numLabel7.textContent = ++num7; 
    
 
// });


btnPredict.addEventListener('click', async function() {
    isPredicting = !isPredicting; 
    if(isPredicting) {
        console.log("If statement working ");
        this.textContent = 'Stop Predicting';
        classifier.predict().then(updatePrediction);
    } else {
        this.textContent = 'start predicting'; 
    }

})


//Recursively allows for predictions to occur 
function updatePrediction(label) {
    lPrediction.textContent = label; 
    console.log('Label is  ' + label); 
    if(isPredicting) {
        classifier.predict().then(updatePrediction);
        }
    }

});


















// let net;
// const webcamElement = document.getElementById('webcam');
// const classifier = knnClassifier.create();


// // async function app() {
// //     console.log('Loading mobilenet..');
  
// //     // Load the model.
// //     net = await mobilenet.load();
// //     console.log('Successfully loaded model');
  
// //     // Create an object from Tensorflow.js data API which could capture image 
// //     // from the web camera as Tensor.
// //     const webcam = await tf.data.webcam(webcamElement);
  
// //     // Reads an image from the webcam and associates it with a specific class
// //     // index.
// //     const addExample = async classId => {
// //       // Capture an image from the web camera.
// //       const img = await webcam.capture();
  
// //       // Get the intermediate activation of MobileNet 'conv_preds' and pass that
// //       // to the KNN classifier.
// //       const activation = net.infer(img, 'conv_preds');
  
// //       // Pass the intermediate activation to the classifier.
// //       classifier.addExample(activation, classId);
  
// //       // Dispose the tensor to release the memory.
// //       img.dispose();
// //     };
  
// //     // When clicking a button, add an example for that class.
// //     document.getElementById('class-a').addEventListener('click', () => addExample(0));
// //     document.getElementById('class-b').addEventListener('click', () => addExample(1));
// //     document.getElementById('class-c').addEventListener('click', () => addExample(2));
  
// //     while (true) {
// //       if (classifier.getNumClasses() > 0) {
// //         const img = await webcam.capture();
  
// //         // Get the activation from mobilenet from the webcam.
// //         const activation = net.infer(img, 'conv_preds');
// //         // Get the most likely class and confidences from the classifier module.
// //         const result = await classifier.predictClass(activation);
  
// //         const classes = ['A', 'B', 'C'];
// //         document.getElementById('console').innerText = `
// //           prediction: ${classes[result.label]}\n
// //           probability: ${result.confidences[result.label]}
// //         `;
  
// //         // Dispose the tensor to release the memory.
// //         img.dispose();
// //       }
  
// //       await tf.nextFrame();
// //     }
// //   }

// // app();

// //Constructor class 
// // Contains the number of classes needed to be predicted
// // declares the variables to hold the mobilenet model and custom model 

// class Classifier {

// constructor(numClasses) {
//     // Storing the number of classes
//     this.numClasses = numClasses;

//     this.modelPath = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
//     this.layerName = '';
//     this.imageSize = 224;

//    this.tag = [];
//    this.samples = {
//        xs: null,
//        ys: null
//    };
// }

// async init () {

//     // Loading original mobilenet model
//     const mobilenet = await tf.loadModel(this.modelPath);

//     //Warming up the model 
//     mobilenet.predict(tf.zeros([1,this.imageSize, this.imageSize, 3])).dispose();

//     //top-slice mobilenet model to use this as a feature extractor
//     const layer = mobilenet.getLayer('conv_pw_13_relu');

//     // return instance to allow for concatenation
//     return this; 
//     }

//     addSample(img, label) {

//         let classID; 

//         if(this.tag.includes(label)) {
//             classID = tags.indexOf(label);
//         } else {
//             classID = this.tags.push(label) -1;   
//         }

//         //Feature extraction
//         const imgTensor  = this.imgToTensor(img);
//         const features = this.mobilenet.predict(img_tensor);

//         // Represents the probability distribution 
//         // let classNo = classID.toint; 
//         const y = tf.tidy(() => tf.oneHot(tf.tensor1d([classID]).toInt(), this.numClasses)); 
//         // add training data and keeping the data using TF.keep 
//         if(this.samples.xs == null) {
//             this.samples.xs = tf.keep(features);
//             this.samples.ys = tf.keep(y);
//         } else {
//             const xs = this.samples.xs;
//             const ys = this.samples.ys;
//             //Concatanating image features and one hot 
//             this.samples.xs = this.keep(xs.concat(img_features, 0));
//             this.samples.ys = tf.keep(ys.concat(y,0));
//             xs.dispose();
//             ys.dispose();
//             y.dispose();
//         }
       
//     }

//     async train(callback, hidedenUnits, learningRate, trainingEpochs) {
//          model = tf.sequential(); 

//          let input = tf.layers.flatten({
//             inputShape: [7,7,256]
//          });
//         let hidden = tf.layers.dense({
//             units: hidedenUnits,
//             activation: 'relu',
//             kernelInitializer: 'varianceScaling',
//             useBias: true,
//         });

//         let outputs  = tf.layers.dense({
//             units: this.numClasses,
//             activation: 'softmax',
//             kernelInitializer: 'varianceScalig',
//             useBias: false,
//         });
//         model.add(input);
//         model.add(hidden);
//         model.add(outputs);

//     }

//     async createModel()
//     {

//     }

// }
