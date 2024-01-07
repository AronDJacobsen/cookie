import os
import torch
from torchvision import datasets, transforms



def process_and_save_data(data_folder, output_folder):

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])  # assuming 0,1 range

    # Load the raw data
    raw_trainset = datasets.FashionMNIST(os.path.join(data_folder, 'raw'), download=True, train=True, transform=transform)
    raw_testset = datasets.FashionMNIST(os.path.join(data_folder, 'raw'), download=True, train=False, transform=transform)

    # Combine train and test data
    combined_data = torch.utils.data.ConcatDataset([raw_trainset, raw_testset])

    # Create a data loader for the combined dataset
    dataloader = torch.utils.data.DataLoader(combined_data, batch_size=64, shuffle=True)

    # Process and normalize the data
    processed_data = []
    for images, labels in dataloader:
        # Your processing steps go here if needed
        # For now, we are just normalizing the data

        # Flatten the images
        flattened_images = images.view(images.size(0), -1)

        # Flatten the labels
        flattened_labels = labels.view(labels.size(0), -1)

        # Normalize the flattened images
        normalized_images = (flattened_images - 0.5) / 0.5

        processed_data.append((normalized_images, flattened_labels))

    # Save the processed data
    processed_data_path = os.path.join(output_folder, 'processed_data.pt')
    torch.save(processed_data, processed_data_path)
    print(f"Processed data saved at: {processed_data_path}")

if __name__ == "__main__":
    # Specify your data and output folders
    data_folder = 'data'
    output_folder = os.path.join(data_folder, 'processed')

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process and save the data
    process_and_save_data(data_folder, output_folder)

    print('Data processed and saved')
