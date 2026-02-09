const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

const modelInfo = {
    version: "v1.0-tfidf-logreg",
    type: "Logistic Regression",
    last_trained: "2026-02-09"
};

app.get('/health', (req, res) => {
    res.status(200).json({ status: "ok", message: "Server is running" });
});

app.get('/model/info', (req, res) => {
    res.status(200).json(modelInfo);
});

app.post('/predict', (req, res) => {
    const { text } = req.body;
    const startTime = Date.now();

    if (!text) {
        return res.status(400).json({ error: "No text provided" });
    }

    // เรียกใช้สคริปต์ Python
    const pythonProcess = spawn('python', ['predict.py', text]);

    let pythonData = "";
    let pythonError = "";

    pythonProcess.stdout.on('data', (data) => {
        pythonData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
        const endTime = Date.now();
        const latency = endTime - startTime;
        
        if (code !== 0) {
            console.error(`Python process exited with code ${code}: ${pythonError}`);
            return res.status(500).json({ error: "Python process error", details: pythonError });
        }

        try {
            const prediction = JSON.parse(pythonData);
            res.json({
                label: prediction.label,
                confidence: prediction.confidence, 
                latency: `${latency} ms`,
                model_version: modelInfo.version
            });
        } catch (e) {
            console.error("JSON Parse Error. Data received:", pythonData);
            res.status(500).json({ error: "Prediction parse failed", details: pythonData });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Backend Server running at http://localhost:${PORT}`);
});    