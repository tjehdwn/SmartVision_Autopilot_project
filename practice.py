import torch
from torch import nn


input_size = 128 
ouput_size = 256
num_layers = 3
bidirectional = True

model = nn.RNN(
    input_size=input_size,
    hidden_size=ouput_size,
    num_layers=num_layers,
    nonlinearity="tanh", #활성화 함수
    batch_first=True,
    bidirectional=bidirectional,
)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size)
h_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size, ouput_size) #초키 은닉층 상태

outputs, hidden = model(inputs, h_0)
print(outputs.shape) #[배치크기, 시퀀스 길이, (양방향+1)x은닉 상태 크기]
print(hidden.shape)