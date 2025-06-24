import torch
import numpy as np
torch.set_printoptions(threshold=torch.inf)
import matplotlib.pyplot as plt, matplotlib.pylab as pylab
import yaml
from sklearn.metrics import mean_squared_error as rmse



def printClassificationResults(dataset_size, prediction_batch, label_batch):
    num_correct = torch.sum(torch.abs(prediction_batch - label_batch) <= 0.5)
    print(f"number correct = {num_correct.item()}/{dataset_size}")

    percent_correct = (num_correct/dataset_size)*100
    print(f"percent correct = {percent_correct.item()}%")


def plotRegressionCurves(ti, Yi, Xi):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(ti, Xi, label='ground truth', color='C0')
    ax2.plot(ti, Yi, label='prediction', color='C1')
    ax1.set_ylabel('ground truth')
    ax2.set_ylabel('prediction')
    rmse_ = rmse(Xi.squeeze(), Yi.squeeze())
    ax1.set_title(f"rmse: {rmse_}" )
    fig.legend()
      

def plotRegressionResults(t, prediction_batch, label_batch, save_dir, show_results):

    for i, (ti, Yi, Xi) in enumerate(zip(t[:5], prediction_batch[:5], label_batch[:5])):
      plotRegressionCurves(ti, Yi, Xi)  
      plt.savefig(f'{save_dir}/regression_{i}.pdf')
      plt.close()

    if show_results:
      for ti, Yi, Xi in zip(t, prediction_batch, label_batch):
        plotRegressionCurves(ti, Yi, Xi)  
        plt.show(block=False)
        plt.pause(1.5)
        plt.close()




def plotTrainingResults(epoch_plt, loss_plt, save_dir):

  epoch_plt = torch.tensor(epoch_plt)
  loss_plt = torch.tensor(loss_plt)
  print(f"mean loss unfiltered = {loss_plt.mean()}")

  plt.figure(1)
  marker_size = 1
  f = plt.scatter(epoch_plt[:], loss_plt[:], s=marker_size)
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.title(f"mean loss = {loss_plt.mean()}")
  plt.grid()


  z = np.polyfit(epoch_plt, loss_plt, 5)
  p = np.poly1d(z)
  pylab.plot(epoch_plt, p(epoch_plt), "r--")


  plt.savefig(f'{save_dir}/loss_plot.pdf')

  plt.show()




if __name__ == "__main__":
  pass
