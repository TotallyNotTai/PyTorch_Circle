import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

def main():
    # values
    num_data_files = 10000
    num_ref_files = 200
    data_x = 30
    data_y = 30
    conv_kernel_size = 3
    conv_layer_size_1 = 4
    conv_layer_size_2 = 16
    hidden_size_1 = 20
    hidden_size_2 = 10
    output_size = 1
    padding_size = 2
    stride_size = 1

    learning_rate = 0.01
    num_epoch = 10
    batch_size = 10

    # import data for training
    class CircleDataSets(Dataset):
        data = torch.zeros(num_data_files, 1, 30, 30, dtype=torch.float32)
        labels = torch.zeros(num_data_files, dtype=torch.float)

        def __init__(self):
            # imports data from the two folders
            for i_1 in range(int(num_data_files/2)):
                with open("./data/generated_training_data/circles/Test_Circles_" + str(i_1 + 1) + ".txt") as f:
                    n_cols = len(f.readline().split(" "))
                data_string = np.loadtxt("./data/generated_training_data/circles/Test_Circles_" + str(i_1 + 1) + ".txt",
                                         delimiter=" ",
                                         usecols=np.arange(0, n_cols - 1),
                                         dtype=np.float32)

                for j in range(data_string.__len__()):
                    self.data[i_1][0][j] = torch.from_numpy(data_string[j])

                self.labels[i_1] = torch.tensor(1)

            for i_2 in range(int(num_data_files/2)):
                with open("./data/generated_training_data/no_circles/Test_Circles_neg_" + str(i_2 + 1) + ".txt") as f:
                    n_cols = len(f.readline().split(" "))
                data_string = np.loadtxt("./data/generated_training_data/no_circles/Test_Circles_neg_" + str(i_2 + 1)
                                         + ".txt",
                                         delimiter=" ",
                                         usecols=np.arange(0, n_cols - 1),
                                         dtype=np.float32)
                for j in range(data_string.__len__()):
                    self.data[i_2+int(num_data_files/2)][0][j] = torch.from_numpy(data_string[j])

                self.labels[i_2+int(num_data_files/2)] = torch.tensor(0)

                self.n_samples = self.labels.__len__()

            self.labels = self.labels.unsqueeze(1)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def __len__(self):
            return self.n_samples

    # import data for testing
    class CircleReferenceDataSets(Dataset):
        data = torch.zeros(num_ref_files, 1, 30, 30, dtype=torch.float32)
        labels = torch.zeros(num_ref_files, dtype=torch.float)

        def __init__(self):
            # imports data from the two files, each assuming only 100 test data each
            for i_1 in range(int(num_ref_files/2)):
                with open("./data/generated_reference_data/circles/Test_Circles_" + str(i_1 + 1) + ".txt") as f:
                    n_cols = len(f.readline().split(" "))
                data_string = np.loadtxt("./data/generated_reference_data/circles/Test_Circles_" + str(i_1 + 1) + ".txt",
                                         delimiter=" ",
                                         usecols=np.arange(0, n_cols - 1),
                                         dtype=np.float32)

                for j in range(data_string.__len__()):
                    self.data[i_1][0][j] = torch.from_numpy(data_string[j])

                self.labels[i_1] = torch.tensor(1)

            for i_2 in range(int(num_ref_files/2)):
                with open("./data/generated_reference_data/no_circles/Test_Circles_neg_" + str(i_2 + 1) + ".txt") as f:
                    n_cols = len(f.readline().split(" "))
                data_string = np.loadtxt(
                    "./data/generated_reference_data/no_circles/Test_Circles_neg_" + str(i_2 + 1) + ".txt",
                    delimiter=" ",
                    usecols=np.arange(0, n_cols - 1),
                    dtype=np.float32)
                for j in range(data_string.__len__()):
                    self.data[i_2 + int(num_ref_files/2)][0][j] = torch.from_numpy(data_string[j])

                self.labels[i_2 + int(num_ref_files/2)] = torch.tensor(0)

                self.n_samples = self.labels.__len__()

            self.labels = self.labels.unsqueeze(1)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def __len__(self):
            return self.n_samples

    # assuming one 2x2 pooling layer, calculating the first linear layer of ConvNeuralNetwork
    def conv_pool_size_calc(data_size, kernel_size, padding_num, stride_num):
        return (((data_size - kernel_size + 2*padding_num) / stride_num) + 1)/2

    final_conv_pool_x = conv_pool_size_calc(conv_pool_size_calc(data_x, conv_kernel_size, padding_size, stride_size),
                                            conv_kernel_size, padding_size, stride_size)

    final_conv_pool_y = conv_pool_size_calc(conv_pool_size_calc(data_y, conv_kernel_size, padding_size, stride_size),
                                            conv_kernel_size, padding_size, stride_size)

    class ConvNeuralNetwork(nn.Module):
        # with init(4, 16, 4, 20, 10, 1, 3)
        def __init__(self, conv_layer_size_1, conv_layer_size_2, conv_kernel_size,
                     hidden_size_1, hidden_size_2, output_size, pad_size, stride_size):
            super(ConvNeuralNetwork, self).__init__()
            self.conv1 = nn.Conv2d(1, conv_layer_size_1, kernel_size=conv_kernel_size, padding=pad_size)  # (1, 4, 4)
            self.conv2 = nn.Conv2d(conv_layer_size_1, conv_layer_size_2, kernel_size=conv_kernel_size,
                                   padding=pad_size, stride=stride_size)
            # (4, 16, 4)
            self.pool = nn.MaxPool2d(2, 2)
            self.l1 = nn.Linear(int(final_conv_pool_x)*int(final_conv_pool_y)*conv_layer_size_2, hidden_size_1)         # (9*9*16, 20)
            self.l2 = nn.Linear(hidden_size_1, hidden_size_2)  # (20, 10)
            self.l3 = nn.Linear(hidden_size_2, output_size)  # (10, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            out = self.pool(F.leaky_relu(self.conv1(x)))                                                                # (1, 4, 4), 32x32x4 -> pool -> 16x16x4
            out = self.pool(F.leaky_relu(self.conv2(out)))                                                              # (4, 16, 4), 18x18x16 -> pool -> 9x9x16
            out = torch.flatten(out, 1)
            out = F.leaky_relu(self.l1(out))
            out = F.leaky_relu(self.l2(out))
            out = self.sig(self.l3(out))
            return out

    circle_data = CircleDataSets()
    print("CircleDataSet loaded")
    test_data = CircleReferenceDataSets()
    print("CircleRefSet loaded")

    # print(circle_data.data)

    # circle_data.data = nn.Flatten()(circle_data.data)
    # test_data.data = nn.Flatten()(test_data.data)

    final_circle_datas = torch.utils.data.DataLoader(dataset=circle_data, batch_size=batch_size, shuffle=True)
    final_test_datas = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    torch.autograd.set_detect_anomaly(True)

    # model
    input_size = data_x*data_y
    model = ConvNeuralNetwork(conv_layer_size_1=conv_layer_size_1, conv_layer_size_2=conv_layer_size_2,
                              conv_kernel_size=conv_kernel_size,
                              hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                              output_size=output_size, pad_size=padding_size, stride_size=stride_size)



    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_step = len(final_circle_datas)
    loss_per_epoch_training = []
    accuracy_per_epoch_training = []
    for epoch in range(num_epoch):
        training_loss = 0
        correct_classified = 0
        total_classified = 0
        for i, (circles, labels) in enumerate(final_circle_datas):
            # circles = circles.view(circles.size(0), -1)

            # print(circles)

            optimizer.zero_grad()

            # forward pass
            pred = model(circles)

            total_classified += labels.size(0)

            values_pred, indices_pred = torch.max(pred, dim=1)
            values_true, indices_true = torch.max(labels, dim=1)
            correct_classified += torch.sum(torch.eq(indices_pred, indices_true)).item()

            loss = loss_fn(pred, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"epoch: {epoch+1}/{num_epoch}, step {i+1}/{n_total_step}, loss = {loss.item():.4f}")

            training_loss += loss.data.item()

        training_loss /= len(final_circle_datas)
        loss_per_epoch_training.append(training_loss)

        print("Epoch " + str(epoch))
        print("Training set loss: " + format(training_loss, '.2f'))
        accuracy = correct_classified / total_classified * 100.0
        accuracy_per_epoch_training.append(accuracy)
        print("Training set accuracy: " + format(accuracy, '.2f') + "%")

#    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        list_outputs = []
        for data, labels in final_test_datas:
            optimizer.zero_grad()

            output = model(data)

            for i in range(len(output)):
                list_outputs.append(float("{0:.3f}".format(output[i].item())))

                # calculate prediction

                if output[i] > 0.9:
                    predictions = 1
                elif output[i] <= 0.1:
                    predictions = 0
                else:
                    predictions = -1

                n_samples += 1

                if predictions == labels[i]:
                    n_correct += 1

        acc = 100.0 * n_correct/n_samples
        print(f'Test set accuracy = {acc}')

        print(list_outputs)

def none():
    pass