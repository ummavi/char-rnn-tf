# char-rnn-tf
A character-level RNN implemented in TensorFlow using a Multi Layer LSTM to capture syntactic structure in data and use it to predict the next character given the sequence it's seen so far. We use this network to learn how to mimic the patterns it sees in the observed data and generate text that best looks like the data.

Code was heavily influenced by https://github.com/sherjilozair/char-rnn-tensorflow


## Usage
To train on a dataset of your choice, simply dump all the information in a .txt file and pass it to the dataset loader with meaningful parameters. Change the required paths in `main.py` and run it. 

You can monitor progress by either looking at the sample generated during training or loading a checkpoint file and generating more samples. This can be done by pointing it to the right checkpoint in `sample.py`


## Results
Note: These results are quite crude and can be significantly improved by longer training times and a more thorough hyperparameter/architecture choice. 

#### Song Lyrics
A dataset of song lyrics obtained from https://www.kaggle.com/mousehead/songlyrics. This consists of songs from over 600 different artists with a total of 57k+ songs from a wide variety of styles and genres. The dataset was first cleaned to remove artist/other meta information and only the lyrics were retained and put into a .txt file. Here are some of the results obtained after 59000 training steps.


>The sleepy streets you heard  
>And the wind was supposed to say  
>The water had a dillary  
>When the stars come to mourn  
>The last was something that I can be  
>I thought we should know what you'd come  
>And I was alright  
>I was a love worder  
>And the last time we were easy  
>And if the day is dead  
>That we have said  
> 
>I love you, now why  
>A part of you  
>I want to be the same  
>  
>I want to take you around  
>I won't say you very hard  
>I want to be your soul  
>I want to be your heart  




>The sound of your soul,  
>People want to think about  
>It plays your mand, tell me that I'm alone  
>I can't help fuch anything to believe it  
>I feel  it's just a little fire  
>There's a place I know it's so  
>I know when you're my life  
>A little for this empty old search  
>I took for the meaning  
>  
>Since I can feel you there  
>To the dear  
>I'm in love with you  
>I need a little bit of myself  
>  
>I won't fall and done  
>So I can see you  
>I want to see  


#### Mark Twain Books
A small and simple dataset was initially used and consisted of two books obtained from Project Gutenburg
* The Adventures of Tom Sawyer by Mark Twain - https://www.gutenberg.org/ebooks/74 
* Adventures of Huckleberry Finn by Mark Twain - https://www.gutenberg.org/ebooks/76

The books were taken and concatenated into one text file then trained by a network with three layers of LSTM cells of 512 each. Here are some of the results:

>Three minutes later the shadow was gone, and the sheet and the same steamboat was soon and the wamer to the steamboat.  In the other side the coffen was so the same way, and the way I come to the widder and the three tow some back and started in the stabboard, and says:
>
>“What did you say your name was?”
>
>“Why, to dig him do it.  I hain’t see any other way.  I was a-gwyne to steal a body to see a boy that was all right now.  I said I would take my hand on the woods and go and see the there they was and the s




>So I’ll mosey along and shouting along and say, “You do torright along the river, and then I come to think you could get all the time.  I wanted to get the best way and the widow and the duke and the duke and the shirt of the bottom of the river, and the whole thing to do in the dark, and I wanted to get a little worse to the town and come to the steamboat and the widow and the widow she would take it out of the woods and the the widow and the duke and the door, and the poor courners was out of sight and the


## Read More
To understand more about what's going on, I highly recommend Andrej Karpathy's blog post on RNNs and a description of the char-rnn (https://karpathy.github.io/2015/05/21/rnn-effectiveness/)