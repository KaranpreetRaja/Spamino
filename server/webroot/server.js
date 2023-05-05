const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();

app.use(express.static('.'));

app.get('/api/files', (req, res) => {
    fs.readdir('.', (err, files) => {
        if (err) {
            console.error(`Error reading directory: ${err}`);
            res.status(500).send('An error occurred while listing files.');
            return;
        }
        const mailFiles = files.filter(file => file.endsWith('.mail'));
        res.json(mailFiles);
    });
});

app.get('/api/files/:filename', (req, res) => {
    const filename = req.params.filename;
    if (!filename.endsWith('.mail')) {
        res.status(404).send('File not found.');
        return;
    }

    fs.readFile(path.join('.', filename), 'utf8', (err, content) => {
        if (err) {
            console.error(`Error reading file: ${err}`);
            res.status(500).send('An error occurred while reading the file.');
            return;
        }
        res.send(content);
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server listening at http://localhost:${PORT}`);
});
