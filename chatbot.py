
# Import libraries we need

import numpy
import tensorflow
import tflearn
import random
import json
import pickle
import nltk

from nltk.stem.lancaster import LancasterStemmer
word_stemmer = LancasterStemmer()

# Read training data from the json file
with open("training.json") as file:
    data = json.load(file)

try:
    # [IMP: If you changing the json input data, delete the pickle file and try running code again]

    # Try loading already pre-processed data from the pickle file.
    with open("cache.pickle", "rb") as f:
        vocabulary, labels, training, output = pickle.load(f)
except:

    # If no stored data in pickle file, pre-process the data, and store it now.
    vocabulary = []
    labels = []
    docs_x = []
    docs_y = []

    # Extract words from the patterns, extract labels (tags)

    for intent in data["intents"]:
        for pattern in intent["patterns"]:

            tokenized_words = nltk.word_tokenize(pattern)
            vocabulary.extend(tokenized_words)
            docs_x.append(tokenized_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stem the words. This process will bring the words to its root (the main meaning)

    vocabulary = [word_stemmer.stem(w.lower()) for w in vocabulary if w != "?"]
    vocabulary = sorted(list(set(vocabulary)))

    labels = sorted(labels)

    # Neural network doesn't understand strings at all so we need to convert our input and
    # output into a list of numbers. This is called "a bag of words" in neural network terminoloy.
    # And the process to covert string into numbers is called "One hot encoding".

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag_of_words = []

        wrds = [word_stemmer.stem(w) for w in doc]

        for w in vocabulary:
            if w in wrds:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag_of_words)
        output.append(output_row)

    # Convert the two lists (training input list) and (output) into numpy arrays because tflearn works
    # with numpy arrays.

    training = numpy.array(training)
    output = numpy.array(output)

    # Write the pre-processed data into the pickle file to use it subsequent time.
    with open("cache.pickle", "wb") as f:
        pickle.dump((vocabulary, labels, training, output), f)


# Resetting previous underline data graphs with the neural network
tensorflow.reset_default_graph()

# Input neural layer with number of neurons equals to the number of words in the training set
neural_network = tflearn.input_data(shape=[None, len(training[0])])

# First hidden layer with 8 neurons
neural_network = tflearn.fully_connected(neural_network, 8)

# Second hidden layer with 8 neurons
neural_network = tflearn.fully_connected(neural_network, 8)

# Output neuron layer with the number of neurons equal to number of tags.
# Softmax is used to give a probability to each of the neurons in the output layer.
neural_network = tflearn.fully_connected(neural_network, len(output[0]), activation="softmax")

# Apply regression to the network
neural_network = tflearn.regression(neural_network)

# Activate Deep Neural network algorithm to train the model
AI_model = tflearn.DNN(neural_network)

# ------------------ Future Optimization [Ignore for now] -----------------------
# try:
#     # If the model is already trained, load the trained one.
#     model.load("model.tflearn")
# except:
#     # Pass training and output data
#     # Number of epochs is the amount of time the model is going to see the same data.
#     model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#
#     # save the model
#     model.save("model.tflearn")
# ------------------ Future Optimization [Ignore for now] -----------------------

# Pass training and output data
# Number of epochs is the amount of time the model is going to see the same data.
AI_model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

# save the model
AI_model.save("model.tflearn")


def bag_of_words(user_query, vocabulary):
    bag = [0 for _ in range(len(vocabulary))]

    s_words = nltk.word_tokenize(user_query)
    s_words = [word_stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(vocabulary):
            if se == w:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Let the model predict
        result = AI_model.predict([bag_of_words(inp, vocabulary)])

        # Choose the result with maximum probability
        result_index = numpy.argmax(result)

        # If the max result is more than 70% probability, then show the answer from the tags.

        #[IMP: If you are not getting good responses, try loweing the accuracy]
        if result[0][result_index] > 0.6:

            # Take the tag corresponding to that result
            tag = labels[int(result_index)]

            responses = []
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))

        else:

            print("I didn't get that. Try again.")

chat()