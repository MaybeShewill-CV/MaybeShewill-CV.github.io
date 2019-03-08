/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const ATTENTIVE_DERAIN_MODEL_PATH ='derain_model/tensorflowjs_model.pb';
const ATTENTIVE_DERAIN_WEIGHTS_PATH ='derain_model/weights_manifest.json';

const IMAGE_WIDTH = 360;
const IMAGE_HEIGHT = 240;

let attentive_derain_net;
const derainnetDemo = async () => {
  status('Loading model...');

  attentive_derain_net = await tf.loadGraphModel(ATTENTIVE_DERAIN_MODEL_PATH, ATTENTIVE_DERAIN_WEIGHTS_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  attentive_derain_net.predict(tf.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])).dispose();

  console.log('Model Warm complete');

  // Make a prediction through the locally hosted test_rain.jpg.
  const image_Element = document.getElementById('test_rain');
  if (image_Element.complete && image_Element.naturalHeight !== 0) {

    predict(image_Element);
    image_Element.style.display = '';
  } else {

    image_Element.onload = () => {
      predict(image_Element);
      image_Element.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};


/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');

  const startTime = performance.now();
  const derain_images = tf.tidy(() => {

    const img = tf.browser.fromPixels(imgElement).toFloat();

    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.div(127.5).sub(1.0);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]);

    // Make a prediction through derain net
    return attentive_derain_net.predict(batched);
  });

  const derain_images_int32 = tf.cast(derain_images, 'int32');

  // Convert logits to probabilities and class names.
  const totalTime = performance.now() - startTime;
  status(`Done in ${Math.floor(totalTime)}ms`);

  // Show the classes in the DOM.
  showResults_Derain_Result(imgElement);
  await tf.browser.toPixels(derain_images_int32, derain_result_canvas);
}

//
// UI
//

function showResults_Derain_Result(srcimgElement) {

  const sourceContainer = document.createElement('div');
  sourceContainer.className = 'src-container';

  const srcImgContainer = document.createElement('div');
  srcImgContainer.appendChild(srcimgElement);

  sourceContainer.appendChild(srcImgContainer);

  sourceImgElement.insertBefore(
      sourceContainer, sourceImgElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      //let img = document.createElement('img');
      let img = document.getElementById('test_rain');
      img.src = e.target.result;
      img.width = IMAGE_WIDTH;
      img.height = IMAGE_HEIGHT;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const sourceImgElement = document.getElementById('SourceImgs');
const derain_result_canvas = document.getElementById('derain_result-canvas');

derainnetDemo();
