import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


def main():
    # values
    num_data_files = 10000
    num_ref_files = 200
    data_x = 30
    data_y = 30
    hidden_size = 150
    output_size = 1
    learning_rate = 0.01
    num_epoch = 5
    batch_size = 1

    # import data for training
    class CircleDataSets(Dataset):
        def __init__(self):
            self.data = torch.empty(0)
            self.labels = torch.empty(0)
            self.__data_list__ = []
            self.__label_list__ = []

            # imports data from single file in folder
            with open("./data/test_small/Test_Circles.txt") as f:
                n_cols = len(f.readline().split(","))
            self.data_string = np.loadtxt("./data/test_small/Test_Circles.txt",
                                          delimiter=",",
                                          usecols=np.arange(0, n_cols - 1),
                                          dtype=np.float32)

            for i in range(self.data_string.__len__()):
                label_number = self.data_string[i][0]

                self.__label_list__.append(self.data_string[i][1:int(label_number + 1)])

                # reshaping the numpy array into fitting shape for later conversion into right size
                self.__data_list__.append(self.data_string[i][int(label_number + 1):].reshape(data_x, data_y))

            self.labels = torch.from_numpy(np.array(self.__label_list__))
            self.data = torch.from_numpy(np.array(self.__data_list__))

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def __len__(self):
            return len(self.labels)

        def input_size(self):
            return data_x * data_y

        def output_size(self):
            return len(self.labels[0])

    # import data for testing
    class CircleReferenceDataSets(Dataset):
        def __init__(self):
            self.data = torch.empty(0)
            self.labels = torch.empty(0)
            self.__data_list__ = []
            self.__label_list__ = []

            # imports data from single file in folder
            with open("./data/test_small/Test_Circles.txt") as f:
                n_cols = len(f.readline().split(","))
            self.__data_string__ = np.loadtxt("./data/test_small/Test_Circles.txt",
                                              delimiter=",",
                                              usecols=np.arange(0, n_cols - 1),
                                              dtype=np.float32)

            for i in range(self.__data_string__.__len__()):
                label_number = self.__data_string__[i][0]

                self.__label_list__.append(self.__data_string__[i][1:int(label_number + 1)])

                # reshaping the numpy array into fitting shape for later conversion into right size
                self.__data_list__.append(self.__data_string__[i][int(label_number + 1):].reshape(data_x, data_y))

            self.labels = torch.from_numpy(np.array(self.__label_list__))
            self.data = torch.from_numpy(np.array(self.__data_list__))

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def __len__(self):
            return len(self.labels)

        def input_size(self):
            return data_x * data_y

        def output_size(self):
            return len(self.labels[0])

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.LeakyReLU()
            self.l2 = nn.Linear(hidden_size, output_size)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            return self.sig(out)

    circle_data = CircleDataSets()
    referencing_data = CircleReferenceDataSets()

    # circle_data.data = nn.Flatten()(circle_data.data)
    # test_data.data = nn.Flatten()(test_data.data)

    final_circle_datas = torch.utils.data.DataLoader(dataset=circle_data, batch_size=batch_size, shuffle=True)
    final_reference_datas = torch.utils.data.DataLoader(dataset=referencing_data, batch_size=batch_size, shuffle=False)

    torch.autograd.set_detect_anomaly(True)

    # model
    input_size = circle_data.input_size()
    output_size = circle_data.output_size()
    model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_step = len(final_circle_datas)
    for epoch in range(num_epoch):
        for i, (circles, labels) in enumerate(final_circle_datas):
            optimizer.zero_grad()
            circles = nn.Flatten()(circles)

            # forward pass
            pred = model(circles)
            loss = loss_fn(pred, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"epoch: {epoch + 1}/{num_epoch}, step {i + 1}/{n_total_step}, loss = {loss.item():.4f}")

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        list_outputs = []
        for data, labels in final_reference_datas:
            data = nn.Flatten()(data)

            output = model(data)

            for i in range(len(output)):
                for j in range(len(output[i])):
                    list_outputs.append(float("{0:.3f}".format(output[i][j])))

                    # calculate prediction

                    if output[i][j] <= 0.1:
                        predictions = 0
                    elif 1.1 >= output[i][j] >= 0.9:
                        predictions = 1
                    elif 2.1 >= output[i][j] >= 1.9:
                        predictions = 2
                    elif 3.1 >= output[i][j] >= 2.9:
                        predictions = 3
                    else:
                        predictions = -1

                    n_samples += 1
                    if predictions == labels[i][j]:
                        n_correct += 1

        acc = 100.0 * n_correct / n_samples
        print(f'accuracy = {acc}')

        print(list_outputs)


def none():
    pass