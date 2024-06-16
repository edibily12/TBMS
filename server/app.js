const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const app = express();
const port = 3000;

const upload = multer({ dest: '../output/' });

app.post('/upload', upload.single('file'), async (req, res) => {
    const file = req.file;

    if (!file) {
        return res.status(400).send('No file uploaded.');
    }

    const formData = new FormData();
    formData.append('file', fs.createReadStream(file.path));

    try {
        const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
            headers: formData.getHeaders()
        });

        const data = response.data;
        if (data.error) {
            return res.status(500).send(data.error);
        }

        const prediction = data.prediction;
        res.json({ prediction });

    } catch (error) {
        console.error('Error making prediction:', error.response ? error.response.data : error.message);
        res.status(500).send('Error making prediction');
    } finally {
        fs.unlinkSync(file.path);
    }
});

app.listen(port, function () {
    console.log("app listening on port " + port);
});
