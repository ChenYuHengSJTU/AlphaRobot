import torch

class DQN(torch.nn.Module):
    def __init__(self, input_dim, action_space):
        super(DQN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        self.feature_size = self.features(
            torch.zeros(1, input_dim , input_dim)).cuda().view(1, -1).size(1)
        
        # print(self.feature_size)
        
        self.value = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_space)
        )

        # self.actor = torch.nn.Sequential(
        #     torch.nn.Linear(feature_size, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, action_space),
        #     torch.nn.Softmax(dim=-1)
        # )
        
        # torch.nn.init.xavier_normal_(self.features.weight.data)
        # torch.nn.init.xavier_normal_(self.value.get_parameters())

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)


    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1, self.feature_size)
        # print(x.shape)
        # print(a.shape)
        
        # x = torch.concat((x, a), dim=-1)
                
        value = self.value(x)
        # actions = self.actor(x)
        return value