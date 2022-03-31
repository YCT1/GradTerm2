from turtle import forward
import torch

class exampleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(2,3)
    
    def forward(self, X):
        return self.layer1(X)
        
