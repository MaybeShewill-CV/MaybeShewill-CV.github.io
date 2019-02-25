const cat = document.getElementById('cat');
cat.onload = async () => {
  const resultElement = document.getElementById('result');

  resultElement.innerText = 'Loading NsfwNet...';

  console.log('Load nsfw model start');

  const nsfwNet = new NsfwNet();
  console.time('Loading of model');
  await nsfwNet.load();
  console.timeEnd('Loading of model');

  console.log('Load nsfw model successfully');

  const pixels = tf.browser.fromPixels(cat);

  console.time('First prediction');
  let result = nsfwNet.predict(pixels);
  const topK = nsfwNet.getTopKClasses(result, 2);
  console.timeEnd('First prediction');

  resultElement.innerText = '';
  topK.forEach(x => {
    resultElement.innerText += `${x.value.toFixed(3)}: ${x.label}\n`;
  });

  console.time('Subsequent predictions');
  result = nsfwNet.predict(pixels);
  nsfwNet.getTopKClasses(result, 5);
  console.timeEnd('Subsequent predictions');

  nsfwNet.dispose();
};
cat.src = imageURL;
