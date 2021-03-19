using System;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace LinearRegressionDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataX = np.array(
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
            var dataY = np.array(
                1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
            var samples = dataX.shape[0];
            
            var X = tf.placeholder(tf.float32);
            var Y = tf.placeholder(tf.float32);
            
            var W = tf.Variable(0.0f, name: "weight");
            var b = tf.Variable(0.0f, name: "bias");
            
            var model = tf.add(tf.multiply(X, W), b);           
            var loss = tf.reduce_sum(tf.pow(model - Y, 2.0f)) / (2.0f * samples);
           
            var epochs = 1000;
            var learningRate = 0.01f;
            var displayEvery = 50;
            
            var optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss);            
            var init = tf.global_variables_initializer();

            using (var sess = tf.Session())
            {
                sess.run(init);
                
                Console.WriteLine("Training model...");
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    foreach (var (x, y) in zip<float>(dataX, dataY))
                    {
                        sess.run(optimizer,
                            new FeedItem(X, x),
                            new FeedItem(Y, y));
                    }
                    
                    if ((epoch + 1) % displayEvery == 0)
                    {
                        var lossValue = sess.run(loss, new FeedItem(X, dataX), new FeedItem(Y, dataY));
                        Console.WriteLine($"  epoch: {epoch + 1}\tMSE = {lossValue}\tW = {sess.run(W)}\tb = {sess.run(b)}");
                    }
                }
                
                var trainingLoss = sess.run(loss,
                    new FeedItem(X, dataX),
                    new FeedItem(Y, dataY));

                Console.WriteLine($"  Final MSE = {trainingLoss}");
            }

            
        }
    }
}
