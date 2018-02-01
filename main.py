import tensorflow as tf
import numpy as np

from graph import char_lstm
from utils import dataset


def generate_random_sample(model_path,num_samples_to_generate = 1):
    """
    Generates a random sample to have a rough idea about how the training is proceeding
    """
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                model = char_lstm({},"test",data_loader.vocab_size)
                sess.run(tf.global_variables_initializer()) 
                saver = tf.train.Saver()
                saver.restore(sess, model_path)
                for i in range(num_samples_to_generate):
                    if num_samples_to_generate>1:
                        print("\n\n******Sample",i,"********")
                    print(model.generate_sample(sess,data_loader.chars,data_loader.vocab))



if __name__ == '__main__':

    #Initialize the data reader, model 
    data_loader = dataset("songs.txt",batch_size=100,seq_length=64)

    # tf.reset_default_graph() ##Only need this if running in notebook.

    model = char_lstm({"batch_size":100,"seq_length":64},"train",data_loader.vocab_size)

    #Training Hook to run the training session
    num_epochs = 200
    save_every = 1000
    print_every = 100
    generate_sample_every = 500

    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    saver = tf.train.Saver(tf.global_variables())
    for e in range(num_epochs):
        state = sess.run(model.initial_state)
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            x, y = data_loader.get_next_batch()
            feed = {model.inputs_t: x, model.targets_t: y}
            for i, (c, h) in enumerate(model.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h

            train_loss, state, _ = sess.run([model.loss, model.final_state, model.train_op], feed)

            #Find actual iter so we can print and save.
            it = e * data_loader.num_batches + b
            if it%print_every==0:
                print("{}/{} (epoch {}), train_loss = {:.3f}"
                          .format(e * data_loader.num_batches + b,
                                  num_epochs * data_loader.num_batches,
                                  e, train_loss))
            if it%save_every==0:
                print("Saving checkpoint as ","model_"+str(it)+".ckpt")
                save_path = saver.save(sess, "model_"+str(it)+".ckpt")

            if it%generate_sample_every==0:
                #Use last generated save_path
                generate_random_sample(save_path) 
    sess.close()
