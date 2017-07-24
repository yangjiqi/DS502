import mxnet as mx


def get_mnist(batch_size):
    """
    Mnxet has embedded mnist dataset, no need to generate .rec file.
    Otherwise it is a must to generate .rec file for customized data.
    Here use NDarray iter to make the mnist iterator

    :param batch_size: how many images per batch
    :return: the iterator of the dataset
    """
    mnist = mx.test_utils.get_mnist()
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    return (train_iter, val_iter)

if __name__ == "__main__":
    # testing
    train, val = get_mnist(100)
    print (train)
    batch_t = train.next()
    d = batch_t.data
    l = batch_t.label
    print ("data", d)
    print (val)