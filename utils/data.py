import torch
from PIL import Image
import numpy as np
import random
torch.set_printoptions(threshold=torch.inf)
import glob, os, re
import matplotlib.pyplot as plt, matplotlib.pylab as pylab
from torchvision import transforms
import yaml



def printInferenceResults(dataset_size, prediction_batch, label_batch):
    print(prediction_batch.shape)
    print(label_batch.shape)

    num_correct = torch.sum(torch.abs(prediction_batch - label_batch) <= 0.5)
    print(f"number correct = {num_correct.item()}/{dataset_size}")

    percent_correct = (num_correct/dataset_size)*100
    print(f"percent correct = {percent_correct.item()}%")


def printPetInferenceResults(dataset_size, img_batch, label_batch, prediction_batch, color_channels, show_images):


  if dataset_size == 1:

    predicted_animal = "dog" if prediction_batch.item() >= 0.5 else "cat"
    animal_label = "dog" if label_batch.item() == 1 else "cat"


    print(f"prediction = {prediction_batch}")
    print(f"predicted animal = {predicted_animal}")

    print(f"truth: {label_batch}")
    print(f"truth label = {animal_label}")


    img_batch = img_batch.cpu()

    plt.imshow(img_batch.squeeze(), cmap='gray')
    plt.title(f'Prediction: {predicted_animal}, Label: {animal_label}')
    plt.show()



  else:

    predictions = ["dog" if pet >= 0.5 else "cat" for pet in prediction_batch.tolist()]
    labels = ["dog" if pet_label == 1 else "cat" for pet_label in label_batch.tolist()]

    print(f"predictions = {predictions}")
    print(f"labels      = {labels}")

    # print(prediction_batch)
    # print(prediction_batch.shape)
    # print(label_batch)

    # num_correct = torch.sum(torch.abs(prediction_batch.squeeze() - label_batch) <= 0.5)
    # print(f"number correct = {num_correct.item()}/{dataset_size}")

    # percent_correct = (num_correct/dataset_size)*100
    # print(f"percent correct = {percent_correct.item()}%")

    printInferenceResults(dataset_size, prediction_batch, label_batch)

    if show_images:
      for (img, pred, lbl) in zip(img_batch, predictions, labels):

        img = img.cpu()

        if color_channels == 1:
          plt.imshow(img.squeeze(), cmap='gray') # grey images

        elif color_channels == 3:
          img = np.transpose(img.numpy(), (1, 2, 0)) # color images
          plt.imshow(img)

        plt.title(f'Prediction: {pred}, Label: {lbl}')

        plt.show(block=False)
        plt.pause(1.5)
        plt.close()





# def printPetInferenceResultsMultiout(dataset_size, img_batch, label_batch, prediction_batch, color_channels):

#   predictions = torch.argmax(prediction_batch, dim=0)

#   truths = torch.argmax(label_batch, dim=0)


#   num_correct =  dataset_size - torch.sum(torch.abs(predictions-truths))
#   percent_correct = (num_correct/dataset_size)*100
#   print(f"number correct = {num_correct.item()}/{dataset_size}")
#   print(f"percent correct = {percent_correct.item()}%")


#   labels = ["dog" if x == 0 else "cat" for x in predictions.tolist()]



#   for (img, pred, lbl) in zip(img_batch, predictions, labels):

#     img = img.cpu()

#     if color_channels == 1:
#       plt.imshow(img.squeeze(), cmap='gray') # grey images

#     elif color_channels == 3:
#       img = np.transpose(img.np(), (1, 2, 0)) # color images
#       plt.imshow(img)

#     plt.title(f'Prediction: {pred}, Label: {lbl}')

#     plt.show(block=False)
#     plt.pause(1.5)
#     plt.close()




def loadPetImageStack(multiout):

  if multiout:
    img_batch = torch.load("petdatatensorsmultiout.pth")
    label_batch = torch.load("pettargettensorsmultiout.pth")
  else:
    img_batch = torch.load("petdatatensors.pth")
    label_batch = torch.load("pettargettensors.pth")

  return img_batch, label_batch









def getPetImageTensor(transform, use):

    label = "dog" if np.random.randint(0, 2) == 1 else "cat"
    if use == "train":
      num = np.random.randint(1, 801)
    else:
      num = np.random.randint(801, 1001)
       

    img = Image.open( f"data/pets/{label}{num}.jpg" )
        
    img.load()
    image = np.asarray( img, dtype="int32" )

    # Load, resize, and transform image
    image_tensor = transform(image)


    return image_tensor, label



def genPetImageStack(dataset_size, use, device_type, img_height, img_width, multi_out, save, color_channels):


    # Create dataset transformation
    transformations = [
        transforms.ToTensor(),
        transforms.Resize((img_height, img_width)),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    # Preallocate tensors for the entire dataset
    data_tensors = torch.zeros((dataset_size, color_channels, img_height, img_width))  # Assuming RGB images
    target_tensors = torch.zeros(dataset_size) if not multi_out else torch.zeros(size=(2, dataset_size))

    for n in range(dataset_size):
        print(f"generating training data.... {(n/dataset_size)*100:.1f}%") if (n % 100) == 0 else None
        # Generate random index and select seed


        if color_channels == 1:
            transformations.append(transforms.Grayscale(num_output_channels=1))

        transform = transforms.Compose(transformations)
        image_tensor, label = getPetImageTensor(transform, use)

        # Insert into preallocated tensor
        data_tensors[n] = image_tensor

        if not multi_out:
            target_tensors[n] = 1 if label == "dog" else 0
        elif multi_out:
            target_tensors[0, n] = 1 if label == "dog" else 0
            target_tensors[1, n] = 1 if label == "cat" else 0


        """Show image"""
        # image_np = np.transpose(image_tensor.numpy(), (1, 2, 0))
        # plt.imshow(image_np)
        # plt.show()

    # if save:
    #     if multi_out:
    #         torch.save(data_tensors, "petdatatensors_multiout.pth")
    #         torch.save(target_tensors, "pettargettensors_multiout.pth")
    #     else:
    #         torch.save(data_tensors, "petdatatensors.pth")
    #         torch.save(target_tensors, "pettargettensors.pth")

    # print(data_tensors.shape)
    # print(target_tensors.shape)
    return data_tensors, target_tensors




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


def fetchMLPParametersFromFile(device_type, directory):

  modelParams = {}

  # Use glob to get all files matching the pattern
  weight_pattern = "layer_*_weights_*.pth"  # Pattern to match
  weight_files = glob.glob(os.path.join(directory, weight_pattern))
  weight_files.sort()

  bias_pattern = "layer_*_biases_*.pth"  # Pattern to match
  bias_files = glob.glob(os.path.join(directory, bias_pattern))
  bias_files.sort()


  for (w_file, b_file) in zip(weight_files, bias_files):

    weights = torch.load(w_file, map_location=device_type)
    biases = torch.load(b_file, map_location=device_type)

    regex_pattern = r"layer_(\d+)_weights_(.*?)\.pth"
    match = re.search(regex_pattern, w_file)

    index = match.group(1)
    activation = match.group(2)

    modelParams.update({f"Layer {index}": [weights, biases, activation, index] })

  return modelParams


def fetchCNNParametersFromFile(device_type, directory):

  modelParams = {}

  # Use glob to get all files matching the pattern
  kernel_pattern = "cnn_layer_*_kernels_*_*_*.pth"  # Pattern to match
  kernel_files = glob.glob(os.path.join(directory, kernel_pattern))
  kernel_files.sort()

  bias_pattern = "cnn_layer_*_biases_*_*_*.pth"  # Pattern to match
  bias_files = glob.glob(os.path.join(directory, bias_pattern))
  bias_files.sort()


  for (k_file, b_file) in zip(kernel_files, bias_files):

    kernels = torch.load(k_file, map_location=device_type)
    biases = torch.load(b_file, map_location=device_type)

    regex_pattern = r"cnn_layer_(\d+)_kernels_(\w+)_([\w]+)_(\d+)\.pth"

    match = re.search(regex_pattern, k_file)

    index = match.group(1)
    activation = match.group(2)
    is_conv = match.group(3)
    stride = match.group(4)

    modelParams.update({f"CNN Layer {index}": [is_conv, kernels, biases, activation, stride, index] })

  return modelParams



def genMatrixStack(n, d=5):
    num_a = n // 2
    num_b = n - num_a

    A = torch.randn(num_a, d, d)
    symmetric_tensors = (A + A.transpose(1, 2)) / 2
    symmetric_labels = torch.ones((num_a, ))

    B = torch.randn(num_b, d, d)
    non_symmetric_labels = torch.zeros((num_b, ))

    ds = torch.cat([symmetric_tensors, B], dim=0)
    labels = torch.cat([symmetric_labels, non_symmetric_labels], dim=0)

    shuffle_indices = torch.randperm(n)

    data_batch = ds[shuffle_indices]
    # print(data_batch)
    data_batch = torch.flatten(data_batch, start_dim=1)
    label_batch = labels[shuffle_indices]
    
    return data_batch, label_batch
   



def load_config(yaml_path):
  with open(yaml_path, "r") as f:
      return yaml.safe_load(f)








if __name__ == "__main__":
  # genPetImageStack(15, 64, 64, False, False, 1)

  # idk = fetchMLPParametersFromFile("cpu", "/params/paramsMLP")
  # idk = fetchCNNParametersFromFile("cpu", "./params/paramsCNN")
  
  idk1, idk2 = genMatrixStack(5, 2)
  print(idk1)
  print(idk2)
