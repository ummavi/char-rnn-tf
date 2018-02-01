import random

import tensorflow as tf
import numpy as np


class char_lstm:
    params = {
        'learning_rate':0.001,
        'batch_size':100,
        'num_layers':3,
        'rnn_size':512,
        'seq_length':64,
        'num_epochs':100,
        'input_keep_prob':0.8,
        'output_keep_prob':0.8,
    }
    
    def __init__(self,params,mode,vocab_size):
        #Overwrite default params with more specific ones passed
        self.params.update(params) 
        self.mode = mode
        self.vocab_size = vocab_size
        
        if self.mode=="test":
            #Since in test we'll run it step by step.
            self.params['batch_size'] = 1
            self.params['seq_length'] = 1

        self.build_graph()
        
    def build_graph(self):
        """Build the computation graph.
        """
        self.is_training_t = tf.constant(self.mode == "train",tf.bool)
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        
        
        self.build_layers()
        self.build_loss()
        
        if self.mode=="train":
            self.build_train_op()
            
    def build_layers(self):
        """Builds the LSTM cells and the inference structure
        """
        
        #Create placeholders for input data
        self.inputs_t = tf.placeholder(
            tf.int32, [self.params['batch_size'], self.params['seq_length']])
        
        self.targets_t = tf.placeholder(
            tf.int32, [self.params['batch_size'], self.params['seq_length']])
    
    
        #Encode the inputs and targets into a one hot vector of vocabulary size
        self.inputs_t_enc = tf.one_hot(self.inputs_t , self.vocab_size ,1.0, 0.0)
        self.targets_t_enc= tf.one_hot(self.targets_t, self.vocab_size, 1.0, 0.0)

        
        def lstm_cell(size):
            return tf.contrib.rnn.LSTMBlockCell(size)
        
        self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.params['rnn_size']) for _ in range(self.params['num_layers']-1)]+[lstm_cell(self.vocab_size)], state_is_tuple=True)

        #Apply Dropout only if it's training.
        if self.mode=="train":
            self.cell = tf.contrib.rnn.DropoutWrapper(self.cell,
                                          input_keep_prob=self.params['input_keep_prob'],
                                          output_keep_prob=self.params['output_keep_prob'])

        self.initial_state = self.cell.zero_state(self.params['batch_size'], tf.float32)
        
             
        inputs = self.inputs_t_enc
        if self.mode=="train":
            inputs = tf.nn.dropout(inputs, self.params['output_keep_prob'])

        
        rnn_outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs, dtype=tf.float32, 
                                            initial_state=self.initial_state) 
        self.final_state = last_state
        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, self.vocab_size])  
        
        self.preds_t = tf.layers.dense(rnn_outputs_flat, self.vocab_size) #Logits
        self.probs_t = tf.nn.softmax(self.preds_t) #Softmax results
        
        
    def build_loss(self):
        """Builds the loss function associated graph components
        """
        self.loss =  tf.nn.softmax_cross_entropy_with_logits(logits = self.preds_t, labels = self.targets_t_enc)
        self.loss =  tf.reshape(self.loss, [self.params['batch_size'], -1])
        
        self.seqloss = tf.reduce_mean(self.loss, 1)
        self.loss = tf.reduce_mean(self.seqloss)


    def build_train_op(self,max_grad_norm=12):
        """Defines the optimizer and the training op.
        Uses clip_by_global_norm for added stability
        """
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        self.train_op = optimizer.apply_gradients( zip(grads, tvars),self.global_step)

    
    def generate_sample(self, sess, chars, vocab, num=256,tau = 1.,start_char=None):
        """Function to generate a sequence step by step. Should be run in test mode.
        
        chars: List of characters used in the vocab
        num: Length of the string sequence to be generated
        tau: Temperature parameter to control "peakiness" of sampling
        start_char: Used to prime the sampling to start with a character for meaningful results
        """
        
        state = sess.run(self.cell.zero_state(1, tf.float32))

        def weighted_pick(weights):
            """Defines a function to do weighted sampling with temprature tau
            """
            normed_weights = np.power(weights,(1./tau))
            normed_weights/=np.sum(normed_weights)
            return np.random.choice(range(len(weights)),p = normed_weights)

        generated_string = random.choice(chars) if start_char is None else start_char
        cur_char = generated_string
        
        #Roll ahead one step at a time
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[cur_char]
            feed = {self.inputs_t: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs_t, self.final_state], feed)
            p = probs[0]
            
            sample = weighted_pick(p)

            pred = chars[sample]
            generated_string += pred
            cur_char = pred
        return generated_string
