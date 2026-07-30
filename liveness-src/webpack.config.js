const path = require('path');

module.exports = {
  entry: './src/Widget.jsx',
  output: {
    filename: 'liveness-widget.js',
    path: path.resolve(__dirname, '..', 'static'),
    library: {
      name: 'LivenessWidget',
      type: 'window',
    },
    publicPath: '/static/',
    clean: false,
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', { targets: 'defaults' }],
              ['@babel/preset-react', { runtime: 'automatic' }],
            ],
          },
        },
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  resolve: {
    extensions: ['.js', '.jsx'],
  },
};
