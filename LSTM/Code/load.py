
import gzip
import cPickle

def readFromPklFile(path):
    f=gzip.open(path,'rb')
    train_set, test_set, target_word2idx, source_idx2vector, source_word2idx, target_idx2vector = cPickle.load(f)
    return train_set, test_set, target_word2idx, source_idx2vector, source_word2idx, target_idx2vector



