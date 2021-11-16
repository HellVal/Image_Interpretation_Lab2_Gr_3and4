import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def vis_loss_curves(in_file, fig_file):
    df = pd.read_csv(in_file).T
    _epoch = np.array(df.index)[1:]
    _train = np.array(df.iloc[:, 0])[1:]
    _valid = np.array(df.iloc[:, 1])[1:]

    plt.ylim((0, 6.5))
    plt.plot(_epoch, _train, c='tomato', label='Training loss')
    plt.plot(_epoch, _valid, c='royalblue', label='Validation loss')

    x_major_locator = plt.MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.title('FCN-Resnet50 MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(fig_file)
    plt.close()


if __name__ == '__main__':
    root = r'C:\ETH Course\Image interpretation\Regression\New_result_2\yatao\v1.0'
    vis_loss_curves(root+r'\log_fcn_resnet50_loss.csv', root+r'\log_fcn_resnet50_loss.png')
