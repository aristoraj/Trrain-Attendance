var path = require('path');
var fs = require('fs');

function mockServer(app) {
  var responseMapper = (url, file, type = 'application/json') => {
    app.use(url, function(req, res) {
      res.setHeader('Content-Type', type);
      res.sendFile(path.join(__dirname, file));
    });
  };
  responseMapper('/api/v1/tickets', 'tickets.json');
}
module.exports = mockServer;
