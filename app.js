const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const path = require('path');

app.use(bodyParser.urlencoded({ extended: false }));

app.set('view engine', 'ejs');
app.set('views', 'views');

app.use('/test', (req, res, next) => {
    const test = req.query.input;
    var spawn = require("child_process").spawn;
    var process = spawn('python', ["./predict.py", test]);

    process.stdout.on('data', function(data) {
        var msg = true;


        if (data.toString().slice(0, 4) == "spam") {
            msg = false;

        }
        res.render('index', {
            popUp: true,
            msgClass: msg
        });

    })


});


app.use('/', (req, res, next) => {
    res.render('index', {
        popUp: false,
        msgClass: false
    });
});

app.listen(3000);