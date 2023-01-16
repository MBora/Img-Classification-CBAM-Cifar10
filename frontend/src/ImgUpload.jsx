import React from 'react'
import { useState } from 'react'
import axios from 'axios'
import styles from './ImgUpload.module.css'

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
    <div className={styles["container"]}> 
        <h1 className={styles["heading"]}>Attention Visualizer</h1>
        <div className={styles["container1"]}>
            {img && <img className={styles["preview-img"]} src={URL.createObjectURL(img)} alt="preview" />}    
            {<img src={attention_map} alt="" className={styles["attnmap"]} />}
            <h1 className={styles["pred"]}>{prediction}</h1>
        </div>
        <div className={styles["container2"]} >
            <input className={styles["input-file"]} type="file" name="image" onChange={handleImage}/>
            <button className={styles["button"]}onClick={handleSubmit}>Submit</button>
        </div>
    </div>

  )
}

export default ImgUpload