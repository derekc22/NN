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


def printRegressionResults(t, prediction_batch, label_batch):
    print(prediction_batch.squeeze().shape)

    for ti, Yi, Xi,in zip(t, prediction_batch, label_batch):
      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      ax1.scatter(ti, Xi, label='ground truth', color='C0')
      ax2.scatter(ti, Yi, label='prediction', color='C1')
      ax1.set_ylabel('ground truth')
      ax2.set_ylabel('prediction')
      rmse_ = rmse(Xi.squeeze(), Yi.squeeze())
      ax1.set_title(f"rmse: {rmse_}" )
      fig.legend()
      plt.show()




def plotTrainingResults(epoch_plt, loss_plt):

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


  plt.savefig('figures/loss_plot.pdf')

  plt.show()






def load_config(yaml_path):
  with open(yaml_path, "r") as f:
    return yaml.safe_load(f)








if __name__ == "__main__":
  pass
