# Quora Insincere
SDML Quora Insincere Challenge
==============================
Quora Insincere is a natural language processing based challenge focused on determining whether a question posted on Quora was sincere or not. To understand what Quora deems insincere visit https://www.kaggle.com/c/quora-insincere-questions-classification/data. One thing to note is that they list several different distinct reasons why a question might be insincere, but we don't need to classify them into different groups, we just have to make the binary decision sincere/insincere. 

## Metric
The metric used for scoring the performance of our models is F1 score. If you are unfamiliar with F1 score here are some resources that can give you an intuitive understanding of why we might use F1 score instead of something else like accuracy.

Video explaining precision, recall and F1: https://www.youtube.com/watch?v=XOgvlpwu0hI
Blog explaining why we need more than accuracy: https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/
Detailed wiki page:https://en.wikipedia.org/wiki/F1_score
Scikit-learn documentation page: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

## Word Representation
One of the most important aspects of this challenge is how we represent the text as numbers. Obviously we can't do any mathematical operations directly on the question "What is Taylor Swift's phone number?" so we need to find some way to represent what a word means numerically. We have lots of options for this so here is a list of resources to explore and evaluate:

Blog explaining count vectorizer, tfidf vectorizer, hashing vectorizer using scikit-learn: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

The youtube playlist I originally cut my teeth on. Covers usage of NLTK for text preprocessing and also a little about word representation and modeling: https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

**I highly recommend grasping the above before jumping into the more advanced techniques shown below.**

Word2Vec: The original paper https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
Gensim, a library that will help you train word2vec embeddings: https://radimrehurek.com/gensim/models/word2vec.html

Blog post explaining the evolution from count vectorizer to word2vec and beyond: https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795

Super cool visualizer of word embeddings: https://projector.tensorflow.org/

There are many other kinds of word embeddings and techniques, but I default to word2vec because I think it is the most approachable to understand. If you want to go down the embedding rabbit hole here is a github with links to all of the major papers and improvements from recent https://github.com/Hironsan/awesome-embedding-models

## Model
Now that we have a way to represent our words as numbers we need to find a model that is able to take that information and turn it into a final 1 or 0 denoting insincere or sincere. One good way to start if you are using count vectorizer, tfidf vectorizer or hashing vectorizer is just going down the list of available in scikit-learn and evaluating their performance. One thing to make note of is the vectorizers return sparse matrices so some models will not work unless you convert them to dense matrices first. https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

Some other models worth evaluating are xgboost and lightgbm. These are both libraries that do variations of gradient boosting. For explanations of gradient boosting: https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/

xgboost and lightgbm usage is very very common on kaggle, but for this specific text based challenge they likely arent the best. 

One major pitfall of all of the previous models is that they arent able to capture the information about the ordering of the words. "The man threw the ball" is very different in meaning to "The ball threw the man" yet these previous methods and models aren't able to capture this. To address this we need to use a model that can capture sequential data. Two possible options are convolutional neural networks's which will slide a window across the words a few at a time or recurrent neural networks's which will look at the elements in order and try to capture the relationship between the words and their order. 

For RNN's there are plenty of kernels available already for this specific competition. Here is one comparing the different embeddings that have been made available to us using a barebones implementation https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

Chris also has a cleaned up baseline kernel https://www.kaggle.com/chriskeown/01-warm-up

One thing that is really important here is that there is a gpu specific version of the RNN implementations, cudnnGRU and cudnnLSTM. Please use these as it's an order of magnitude faster than the none gpu specific version

For CNN's there has been less work, but Christof (one of my teammates) put together a public kernel that is a decent implementation of these https://www.kaggle.com/christofhenkel/inceptioncnn-with-flip. There is a lot of room for research and improvement here because everyone has seemed to focus on the RNN's. 

## Room for improvement
Now that we have covered all of the basics of the competition it's time to look at the areas to tune and make things even better. 

One of the areas that has gotten a lot of attention is in the preprocessing of the text. The text we received is relatively clean, but there still might be some errant foreign characters or special symbols that prevent us from capturing all of the information we could. Christof has another very good kernel where he shows how to do some preprocessing https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings. 

The main takeaway from this is that the embeddings we received for this competition have a finite vocabulary and sometimes that vocabulary doesnt cover the words we actually run in to. For example before an embedding was trained they may have lowercased all of the words so when we look up an uppercase word from our vocabulary in the embedding nothing will show up so we can just try lowercasing it and checking again. One valuable tool when doing this preprocessing is regular expressions https://regexone.com/ is a good place to read up and see exactly what you can do with regex and https://regexr.com/ is a nice sandbox to try out different patterns. 

We built on Chris's kernel a while ago in order to do some analysis of these preprocessings https://www.kaggle.com/ryches/01-warm-up

The goal of this was to look at the examples with the highest loss and compare the original text, the preprocessed text and then the text that actually reached the model after embeddings. This would allow us to do error analysis and see potentially why our model was doign so poorly on those examples because we could see if we are potentially throwing away valuable information in preprocessing or if we are losing words simply because we dont have embeddings for them. 

One other areas of improvement is ensembling, training multiple models and then using them in conjuction. There are many different ways to do this, but do to time constraints (we only have 2 hours for this competition) we are pretty much limited to linear regression or averaging. Here is a tutorial from Marios, one of the top all time kagglers, http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/. He has used ensembles to great effect in many competitions. 

## Conclusion
So far that is our current understanding of the problem and NLP in general. We will keep updating this page week after week with our current findings and any valuable resources we gather. We have a trello board setup to map out the things people are working on https://trello.com/invite/b/yPFmiGL1/acab80af7c042e85a30fa56db1281972/quora-insincere-challenge

And a slack channel to bounce ideas around https://join.slack.com/t/sdmachinelearning/shared_invite/enQtNDk0MTcwODIxNzE5LTYxN2RkMmRjZGEyN2IyMTIzMTM4MGY3OTkyOGFjMmQzZWQyNWExZmQ0NTc5Zjk4MTYzYTE3YWZhNDRmZGE5MWQ


