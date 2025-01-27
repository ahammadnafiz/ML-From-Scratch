import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!file) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      setPrediction(result.class);
    } catch (error) {
      console.error("Prediction failed:", error);
      setPrediction("Error predicting image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Cat/Dog Classifier</h1>
      <form onSubmit={handlePredict}>
        <input 
          type="file" 
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])} 
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Analyzing...' : 'Predict'}
        </button>
      </form>
      {prediction && <h2>This is a {prediction}!</h2>}
    </div>
  );
}

export default App;