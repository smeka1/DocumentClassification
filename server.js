'use strict'
const express = require('express');
const app = express();
const http = require('http');

app.get('/', async(req,res) => {
	console.log('Got a Request');
	res.send('Welcome to Sriramtej Meka\'s classifier');
});

app.listen(3000, () => console.log('Server running on port 3000'));
