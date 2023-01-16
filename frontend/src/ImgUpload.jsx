import React from 'react'
import { useState } from 'react'
import axios from 'axios'

const ImgUpload = () => {
    const [img, setImg] = useState(null) // change the initial value to null
    const handleImage = (e) => {
        console.log(e.target.files[0])
        setImg(e.target.files[0])

    }
    const [prediction, setPrediction] = useState(null)
    const [attention_map, setAttention_map] = useState(null)

    const handleSubmit = () => {
        const formData = new FormData()
        formData.append('image', img) // 'image' is the name of the key in the backend
        //img is the name of the variable that holds the image
        axios.post('http://localhost:8000/predict', formData).then(res => {
            console.log(res.data)
            // setPrediction(res.data)
            setPrediction(res.data.prediction)
            setAttention_map(res.data.attention_map)
        })
    }
  return (
    <div>
        <h1>Image Upload</h1>
        <input type="file" name="image" onChange={handleImage}/>
        {img && <img src={URL.createObjectURL(img)} alt="preview" />}
        <button onClick={handleSubmit}>Submit</button>
        {/* attention is a image */}
        {<img src={attention_map} alt="attention_map" />}
        <h1>{prediction}</h1>
    </div>

  )
}

export default ImgUpload