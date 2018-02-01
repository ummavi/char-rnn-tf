import tensorflow as tf
import numpy as np

from graph import char_lstm
from utils import dataset

if __name__ == '__main__':
    model_path = "Lyrics/model_59000.ckpt"
    num_samples_to_generate = 20
    tau = 0.6
    #Specify a list of characters to begin sampling from. This allows more meaningful starts
    #than randomly starting with a lower case "x" which will lead to rather strange results.
    start_char = np.random.choice(["T","I","J","F","Q","W","S"])

    #Initialize the data reader, model 
    data_loader = dataset("songs.txt",batch_size=100,seq_length=64)

    sess = tf.Session()
    model = char_lstm({},"test",data_loader.vocab_size)
    sess.run(tf.global_variables_initializer()) 
    saver = tf.train.Saver()
    saver.restore(sess, model_path)


    for i in range(num_samples_to_generate):
        print("\n******Sample",i,"********")
        print(model.generate_sample(sess,data_loader.chars,data_loader.vocab,tau=tau,start_char=start_char,num=512))
