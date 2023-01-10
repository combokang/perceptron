# %%
from pandas import DataFrame, read_csv
from random import uniform
import matplotlib.pyplot as plt

# perceptron training function


def train_AND_perceptron(data: DataFrame, learning_rate: float) -> tuple:
    '''_
    data: input data    
    learning_rate: speed of learning
    '''
    # initialize and keep weights
    w0 = uniform(-1, 1)
    w1 = uniform(-1, 1)
    w2 = uniform(-1, 1)

    # assign a list space for weights
    weights = [[w0, w1, w2] for i in data.index]

    # loss function: square error
    # assign a list space for loss
    loss = []

    # start training
    print("epoch\tinput\tdesired\tactual\tweights")
    epoch = 0
    while True:
        epoch += 1
        pattern_loss_acc = 0
        # for each epoch
        for index in data.index:
            # for each training record
            record = data.loc[index].values
            x0 = 1  # bias
            x1 = record[0]  # input 1
            x2 = record[1]  # input 2
            y = record[2]   # desired output

            # calculate sumproduct and transform to signal
            sumproduct = x0*w0+x1*w1+x2*w2
            if sumproduct > 0:
                s = 1
            elif sumproduct < 0:
                s = 0
            else:   # cannot get signal when sumproduct == 0, assgin s to the opposit of y
                s = 1-y

            # adjust weight
            if sumproduct == 0 or s != y:
                if y == 1:
                    w0 += learning_rate*x0
                    w1 += learning_rate*x1
                    w2 += learning_rate*x2
                if y == 0:
                    w0 -= learning_rate*x0
                    w1 -= learning_rate*x1
                    w2 -= learning_rate*x2

            # calculate the accumulation of pattern loss
            pattern_loss_acc += ((s-y)**2)/2

            # keep weights
            weights[index] = [w0, w1, w2]
            print(
                f"{epoch}\t{x1,x2}\t{y}\t{s}\t{w0, w1, w2}")

        # calculate epoch loss
        loss.append(pattern_loss_acc/len(training_data.index))

        # if the weights of a epoch are convergent then stop training
        result = True
        first_element = weights[0]
        for w in weights:
            if first_element != w:
                result = False
                break
        if result:
            print("final weights:", weights[index])
            return weights[index], loss


# perceptron perdicting function
def AND_perceptron_perdiction(data: DataFrame, weights: list) -> list:
    '''_
    data: input data    
    weights: trained weights
    '''
    # set weights
    w0, w1, w2 = weights

    # assign a list space for outputs
    outputs = []

    # start perdiction
    for index in data.index:
        # for each testing record
        record = data.loc[index].values
        x0 = 1  # bias
        x1 = record[0]  # input 1
        x2 = record[1]  # input 2

        # calculate sumproduct and transform to signal
        sumproduct = x0*w0+x1*w1+x2*w2
        if sumproduct > 0:
            s = 1
        elif sumproduct < 0:
            s = 0
        else:   # cannot get signal when sumproduct == 0
            s = -1

        outputs.append(s)
        print(f"input:{x1,x2}, output:{s}")

    return outputs


# %%
# read training data
training_data = read_csv("AND.csv")

final_weights, loss = train_AND_perceptron(training_data, 1.0)

# %%
epoch_list = [i for i in range(1, len(loss)+1)]
plt.plot(epoch_list, loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %%
testing_data = training_data
AND_perceptron_perdiction(testing_data, final_weights)

# %%
