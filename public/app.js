const dataSet = tf.data.csv('./kc_house_data.csv')

dataSet.take(10).toArray()
  .then(result => console.log(result))
