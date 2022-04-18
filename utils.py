import pickle


def load(filename):
    with open(filename, 'rb+') as file:
        return pickle.load(file)


def save(obj, filename, type='pickle'):
    if type == 'pickle':
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=4)
