import mxnet as mx
import numpy as np
import cv2
import logging
from data import get_mnist
from mlp_sym import get_mlp_sym
from mlp_sym import get_conv_sym
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

if __name__=="__main__":

    # Get the data iterator
    batch_size = 100
    train_iter, val_iter = get_mnist(batch_size)

    # Get symbol
    # model = get_mlp_sym()
    # todo: model = get_conv_sym()
    model = get_conv_sym()

    # Viz the graph and save the plot for debugging
    plot = mx.viz.plot_network(model, title="mlp", save_format="pdf", hide_weights=True)
    plot.render("MLP")

    # create a trainable module on CPU/GPU
    mod = mx.mod.Module(symbol=model, context=mx.cpu())
    mod.fit(train_iter,  # train data
            eval_data=val_iter,  # validation data
            optimizer='sgd',  # use SGD to train
            optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
            eval_metric='acc',  # report accuracy during training
            batch_end_callback=mx.callback.Speedometer(batch_size, 100),
            # output progress for each 100 data batches
            num_epoch=10)  # train for at most 10 dataset passes

