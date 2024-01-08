import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-react-native'
import { bundleResourceIO } from '@tensorflow/tfjs-react-native'
import React, { useEffect, useState } from 'react'
import { View, Text } from 'react-native'

const modelJSON = require('./assets/model.json')
const modelWeights = require('./assets/group1-shard1of1.bin')

export default function App() {
  const [model, setModel] = useState()

  // 2. Create Recognizer
  const loadModel = async () => {
    const model = await tf.loadGraphModel(bundleResourceIO(modelJSON, modelWeights)).catch((e) => {
      console.log('[LOADING ERROR] info:', e)
    })
    if (!model) return
    // loaded => [{"dtype": "float32", "name": "x", "shape": [-1, 16000]}]
    console.log('loaded =>', model.inputs)
    setModel(model)

    // test
    const inputTensor = tf.randomNormal([1, 16000])
    const prediction = model.predict(inputTensor)
    console.log('prediction', prediction)
  }

 
  useEffect(() => {
    tf.ready()
      .then(() => {
        console.log('[LOADING] tf.ready()')
        loadModel()
      })
      .catch((e) => {
        console.log('[LOADING ERROR] info:', e)
      })
  }, [])


  return <View style={{paddingVertical: 100, paddingHorizontal: 16, display: 'flex', gap: 20}}>
    <Text style={{fontSize: 55}}>Test APP for tensorflowjs</Text>
    <Text style={{fontSize: 35}}>Thank you for helping ! ❤❤❤</Text>
    <Text style={{fontSize: 25, color: "red"}}>Error: Argument 'x' passed to 'gather' must be numeric tensor, but got string tensor</Text>
  </View>
}
