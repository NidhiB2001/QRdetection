// based on https://codelabs.developers.google.com/codelabs/tensorflowjs-object-detection

const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const enableWebcamButton = document.getElementById("webcamButton");

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will
// define in the next step.
if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start classification.
function enableCam(event) {
  // Only continue if the tflite model has finished loading.
  // console.log("model loading finished!!", model)
  if (!model) {
    return;
  }
  if (!qrmodel){
    return;
  }
  // Hide the button once clicked.
  event.target.classList.add("removed");

  // getUsermedia parameters to force video but not audio.
  const constraints = {
    // for back camera in mobile...
    video: { facingMode: 'environment' }            
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    console.log("STREAM DATA");
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// Store the resulting model in the global scope of our app.
var qrmodel = undefined;

tflite.ObjectDetector.create(
  "qrmodel.tflite",
).then((loadedModel) => {
  console.log("loadedModel=========", loadedModel)
  qrmodel = loadedModel;
  // Show demo section now model is ready to use.
  demosSection.classList.remove("invisible");
});

var model = undefined;

tflite.ObjectDetector.create(
  "model.tflite",
).then((loadedModel) => {
  console.log("loadedModel=========", loadedModel)
  model = loadedModel;
  // Show demo section now model is ready to use.
  demosSection.classList.remove("invisible");
});

var qrchildren = [];

function predictWebcam() {
  const qrpredictions = qrmodel.detect(video);

  // Remove any highlighting we did previous frame.
  for (let i = 0; i < qrchildren.length; i++) {
    liveView.removeChild(qrchildren[i]);
  }
  qrchildren.splice(0);

  // Now lets loop through predictions and draw them to the live view if
  // they have a high confidence score.
  // console.log("qrpredictions" , qrpredictions.length);
  for (let i = 0; i < qrpredictions.length; i++) {
    const curqrObject = qrpredictions[i];

    if (curqrObject.classes[0].probability > 0.9) {
    // console.log("CURRENT OBJ", curqrObject.boundingBox);
      const p = document.createElement("p");
      p.innerText =
        curqrObject.classes[0].className +
        " - with " +
        Math.round(parseFloat(curqrObject.classes[0].probability) * 100) +
        "% confidence.";
      p.style =
        "margin-left: " +
        curqrObject.boundingBox.originX +
        "px; margin-top: " +
        (curqrObject.boundingBox.originY - 10) +
        "px; width: " +
        (curqrObject.boundingBox.width - 10) +
        "px; top: 0; left: 0;";

      const qrhighlighter = document.createElement("div");
      qrhighlighter.setAttribute("class", "qrhighlighter");
      qrhighlighter.style =
        "left: " +
        curqrObject.boundingBox.originX +
        "px; top: " +
        curqrObject.boundingBox.originY +
        "px; width: " +
        curqrObject.boundingBox.width +
        "px; height: " +
        curqrObject.boundingBox.height +
        "px;";

      liveView.appendChild(qrhighlighter);
      liveView.appendChild(p);
      qrchildren.push(qrhighlighter);
      qrchildren.push(p);
      captureqr(curqrObject)
    }
  }
  // Call this function again to keep predicting when the browser is ready.
  window.requestAnimationFrame(predictWebcam);
}

var children = [];

function predictWebcam() {
  const predictions = model.detect(video);

  // Remove any highlighting we did previous frame.
  for (let i = 0; i < children.length; i++) {
    liveView.removeChild(children[i]);
  }
  children.splice(0);

  // Now lets loop through predictions and draw them to the live view if
  // they have a high confidence score.
  // console.log("predictions" , predictions.length);
  for (let i = 0; i < predictions.length; i++) {
    const curObject = predictions[i];

    // console.log("curObject" , curObject);
    if (curObject.classes[0].probability > 0.9) {
      // console.log("CURRENT OBJ", curObject.boundingBox);
      const p = document.createElement("p");
      p.innerText =
        curObject.classes[0].className +
        " - with " +
        Math.round(parseFloat(curObject.classes[0].probability) * 100) +
        "% confidence.";
      p.style =
        "margin-left: " +
        curObject.boundingBox.originX +
        "px; margin-top: " +
        (curObject.boundingBox.originY - 10) +
        "px; width: " +
        (curObject.boundingBox.width - 10) +
        "px; top: 0; left: 0;";

      const highlighter = document.createElement("div");
      highlighter.setAttribute("class", "highlighter");
      highlighter.style =
        "left: " +
        curObject.boundingBox.originX +
        "px; top: " +
        curObject.boundingBox.originY +
        "px; width: " +
        curObject.boundingBox.width +
        "px; height: " +
        curObject.boundingBox.height +
        "px;";

      liveView.appendChild(highlighter);
      liveView.appendChild(p);
      children.push(highlighter);
      children.push(p);
      capture(curObject)
    }
  }
  // Call this function again to keep predicting when the browser is ready.
  window.requestAnimationFrame(predictWebcam);
}

function capture(obj) {

  let canvas = document.querySelector("#canvas");
      canvas.getContext('2d').drawImage(video, obj.boundingBox.originX, obj.boundingBox.originY, obj.boundingBox.width, obj.boundingBox.height, 0,0, canvas.width, canvas.height);
      let image_data_url = canvas.toDataURL('image/jpeg');

      // data url of the image
      console.log("cropped image[[[[[]]]]]]",image_data_url);
      // callAPI(image_data_url)
}

function captureqr(obj) {

  let canvasqr = document.querySelector("#canvasqr");
      canvasqr.getContext('2d').drawImage(video, obj.boundingBox.originX, obj.boundingBox.originY, obj.boundingBox.width, obj.boundingBox.height, 0,0, canvasqr.width, canvasqr.height);
      let qrimage_data_url = canvasqr.toDataURL('image/jpeg');

      // data url of the image
      console.log("cropped qrimage[[[[[]]]]]]",qrimage_data_url);
      // callAPI(qrimage_data_url)
}

function callAPI(b64Data) {
// Base64 to blob converter____________
const byteCharacters = atob(b64Data.replace(/^data:image\/(png|jpeg|jpg);base64,/, ''));
const byteNumbers = new Array(byteCharacters.length);
for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
}
const byteArray = new Uint8Array(byteNumbers);
const blob = new Blob([byteArray], {type: 'image/jpeg'});
const file = new File([blob], `crop_${new Date().getTime()}`);
  let formData = new FormData()
  formData.append('file', file)

  fetch("http://127.0.0.1:8011/input",{
      body: formData,
      method: "post"
    });
}