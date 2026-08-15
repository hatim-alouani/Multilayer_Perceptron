import sys
import numpy as np
import helpers

def main():
    model = np.load('saved_model.npy', allow_pickle=True).item()
    helpers.weights = model['weights']
    helpers.biases = model['biases']
    x_min = model['x_min']
    x_max = model['x_max']

    x_valid, y_valid = helpers.load('data_valid.csv')

    x_valid = (x_valid - x_min) / (x_max - x_min)
    y_binary = np.array([1 if v == 'M' else 0 for v in y_valid])

    prediction, l_list = helpers.forward(x_valid)
    p_malignant = prediction[:, 1]

    print('loss: %.4f' % helpers.binary_cross_entropy(y_binary, p_malignant))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
