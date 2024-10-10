import React, { useState } from 'react';
import ReactDOM from 'react-dom';

function App() {
    const [file, setFile] = useState(null);
    const [text, setText] = useState('');

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = () => {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => setText(data.text));
    };

    return (
        <div className="container">
            <h1>Аудио в текст</h1>
            <input type="file" onChange={handleFileChange} />
            <button className="upload-button" onClick={handleUpload}>Загрузить аудио</button>
            <p>Текст: {text}</p>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));
