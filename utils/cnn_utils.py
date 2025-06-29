import torch
from PIL import Image
import numpy as np
torch.set_printoptions(threshold=torch.inf)
import glob, os, re
import matplotlib.pyplot as plt, matplotlib.pylab as pylab
from torchvision import transforms
from utils.data import print_classification_results

def print_pet_inference_results(dataset_size, img_batch, label_batch, prediction_batch, color_channels, show_images):


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

        print_classification_results(dataset_size, prediction_batch, label_batch)

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





# def print_pet_inference_resultsMultiout(dataset_size, img_batch, label_batch, prediction_batch, color_channels):

#     predictions = torch.argmax(prediction_batch, dim=0)

#     truths = torch.argmax(label_batch, dim=0)


#     num_correct =  dataset_size - torch.sum(torch.abs(predictions-truths))
#     percent_correct = (num_correct/dataset_size)*100
#     print(f"number correct = {num_correct.item()}/{dataset_size}")
#     print(f"percent correct = {percent_correct.item()}%")


#     labels = ["dog" if x == 0 else "cat" for x in predictions.tolist()]



#     for (img, pred, lbl) in zip(img_batch, predictions, labels):

#         img = img.cpu()

#         if color_channels == 1:
#             plt.imshow(img.squeeze(), cmap='gray') # grey images

#         elif color_channels == 3:
#             img = np.transpose(img.np(), (1, 2, 0)) # color images
#             plt.imshow(img)

#         plt.title(f'Prediction: {pred}, Label: {lbl}')

#         plt.show(block=False)
#         plt.pause(1.5)
#         plt.close()




def load_pet_img_stack(multiout):

    if multiout:
        img_batch = torch.load("petdatatensorsmultiout.pth")
        label_batch = torch.load("pettargettensorsmultiout.pth")
    else:
        img_batch = torch.load("petdatatensors.pth")
        label_batch = torch.load("pettargettensors.pth")

    return img_batch, label_batch









def get_pet_img_tensor(transform, use):

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



def gen_pet_img_stack(dataset_size, use, device_type, img_height, img_width, multi_out, save, color_channels):


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
        image_tensor, label = get_pet_img_tensor(transform, use)

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




def fetch_cnn_params_from_file(device_type, directory):

    model_params = {}

    # Use glob to get all files matching the pattern
    kernel_pattern = "cnn_layer_*_kernels_*_*_*.pth"  # Pattern to match
    kernel_files = glob.glob(os.path.join(directory, kernel_pattern))
    kernel_files.sort()

    bias_pattern = "cnn_layer_*_biases_*_*_*.pth"  # Pattern to match
    bias_files = glob.glob(os.path.join(directory, bias_pattern))
    bias_files.sort()


    regex_pattern = r"cnn_layer_(\d+)_kernels_(\w+)_([\w]+)_(\d+)\.pth"

    for (k_file, b_file) in zip(kernel_files, bias_files):

        kernels = torch.load(k_file, map_location=device_type)
        biases = torch.load(b_file, map_location=device_type)

        match = re.search(regex_pattern, k_file)

        index = match.group(1)
        activation = match.group(2)
        layer_type = match.group(3)
        stride = match.group(4)

        model_params.update({f"CNN Layer {index}": [layer_type, kernels, biases, activation, stride, index] })

    return model_params