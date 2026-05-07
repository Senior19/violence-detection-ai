# test_dataset.py

from dataset_loader import ViolenceDataset

dataset = ViolenceDataset("dataset/train")

print("Total samples:", len(dataset))

sample, label = dataset[0]

print("Shape:", sample.shape)
print("Label:", label)