let path = require('path');

let projectRootDir = process.cwd();
let sourceFolder = 'src';
let outputFolder = 'app';
let outputFileName = 'extension.js';

module.exports = (mode = 'production') => ({
  entry: path.join(projectRootDir, sourceFolder, 'index.js'),
  output: {
    filename: 'js/' + outputFileName,
    chunkFilename: 'js/[name].js',
    path: path.join(projectRootDir, outputFolder),
    publicPath: './'
  },
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.js$/,
        loader: 'babel-loader',
        options: {
          presets: [
            require.resolve('@babel/preset-env'),
            require.resolve('@babel/preset-react')
          ],
          cacheDirectory: true
        },
        include: path.join(projectRootDir, sourceFolder)
      },
      {
        test: /\.css$/,
        use: [
          'style-loader',
          {
            loader: 'css-loader',
            options: {
              modules: true
            }
          }
        ]
      },
      {
        test: /\.jpe?g$|\.gif$|\.png$/,
        use: ['url-loader?limit=1000&name=./img/[name].[ext]']
      },
      {
        test: /\.woff2|\.woff$|\.ttf$|\.eot$/,
        use: ['url-loader?limit=1000&name=./fonts/[name].[ext]']
      },
      {
        test: /\.svg$/,
        use: ['url-loader?limit=1&name=./fonts/[name].[ext]']
      }
    ]
  }
});