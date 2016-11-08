from mnist import * 
import pickle

if __name__ == "__main__":
    data = read_data_sets(None)
    with open('mnist.pickle','wb') as f:
        pickle.dump(data,f, pickle.HIGHEST_PROTOCOL)
