const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const path = require('path');

app.use(bodyParser.urlencoded({ extended: false }));



app.use('/test', (req, res, next) => {
    const test = req.query.comment;
    var spawn = require("child_process").spawn;
    var process = spawn('python', ["./predict.py", test]);

    process.stdout.on('data', function(data) {
        console.log(data);
        res.send(data.toString());
    })
});


app.use('/', (req, res, next) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

app.listen(3000);