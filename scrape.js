var myArgs = process.argv.slice(2);

var gplay = require('google-play-scraper');

gplay.reviews({
  appId: myArgs[0],
  page: myArgs[1],
  throttle: 10,
}).then(console.log, console.log);

