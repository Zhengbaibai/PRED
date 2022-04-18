<<<<<<< HEAD
import pickle


def load(filename):
    with open(filename, 'rb+') as file:
        return pickle.load(file)


def save(obj, filename, type='pickle'):
    if type == 'pickle':
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=4)
=======
import pickle


def load(filename):
    with open(filename, 'rb+') as file:
        return pickle.load(file)


def save(obj, filename, type='pickle'):
    if type == 'pickle':
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=4)
>>>>>>> fad0c2c7d5f1bd28b868c1169bc7a25da0b518c5
