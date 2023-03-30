import torch
import torch.nn as nn
from collections import namedtuple

class DropoutPart(nn.Module):
    def __init__(self, p, embedding_size):
        super().__init__()
        self.dropout = nn.Dropout(p, inplace=True)
        self.embedding_size = embedding_size

    def forward(self, x):
        self.dropout(x[:, self.embedding_size:])
        return x


class PoseBasicDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 frame_size,
                 after_compression_flat_size=1024, 
                 p_dropout=0.2, 
                 use_dropout=False,
                 hidden_size=[256, 256], 
                 num_of_outputs=4, 
                 action_space=3, 
                 use_act_embedding=False, 
                 use_collision_embedding=False,
                 emb_layers=2,
                 embedding_size=8):
        
        super().__init__()

        num_compression_channels = int(after_compression_flat_size / (frame_size[0] * frame_size[1]))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,  num_compression_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
            nn.Flatten(1),
        )

        self.act_emb = nn.Embedding(action_space, embedding_size) if use_act_embedding else nn.Identity()
        self.colli_emb = nn.Embedding(2, embedding_size) if use_collision_embedding else nn.Identity()
        self.num_of_outputs = num_of_outputs
        self.embedding_size = (use_act_embedding + use_collision_embedding) * embedding_size
        self.use_act_embedding = use_act_embedding
        self.use_collision_embedding = use_collision_embedding
        use_embedding = use_act_embedding | use_collision_embedding
        self.emb_layers = emb_layers

        hidden_size.insert(0, after_compression_flat_size)
        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Sequential(
                DropoutPart(p_dropout, embedding_size) if use_embedding and i < emb_layers else nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i] + self.embedding_size, hidden_size[i + 1]) \
                    if use_embedding and i < emb_layers else nn.Linear(hidden_size[i], hidden_size[i + 1]),
                nn.ReLU(True),
            ))

        self.fc = nn.Sequential(*layers)

        self.p_delta = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size[-1], num_of_outputs)
        )

    def _extract(self, x, action=None, collision=None):

        x = self.conv(x)
        if self.embedding_size != 0:
            for i in range(self.emb_layers):
                _input = []
                if self.use_act_embedding and action is not None:
                    _input.append(self.act_emb(action - 1))
                if self.use_collision_embedding and collision is not None :
                    _input.append(self.colli_emb(collision))
                _input.append(x)

                # for v in _input:
                #     print(v.shape)
                
                x = torch.concat(_input, dim=-1)
                x = self.fc[i](x)
        else:
            x = self.fc(x)

        return x

    def forward(self, x, action=None, collision=None):

        x = self._extract(x, action, collision)
        p_delta = self.p_delta(x)

        out = {'p_delta': p_delta}

        return out

class PoseLaplacianUnerDecoder(PoseBasicDecoder):
    def __init__(self, in_channels, frame_size,
                 after_compression_flat_size=1024, 
                 p_dropout=0.2, use_dropout=False,
                 hidden_size=[256, 256], num_of_outputs=4, 
                 action_space=3, use_act_embedding=False, 
                 use_collision_embedding=False,
                 emb_layers=2,
                 embedding_size=8):
        super().__init__(in_channels, frame_size,
                        after_compression_flat_size, 
                        p_dropout, use_dropout,
                        hidden_size, num_of_outputs, 
                        action_space, use_act_embedding, 
                        use_collision_embedding,
                        emb_layers,
                        embedding_size)
       
        # self.p_sigma = nn.Sequential(
        #     nn.Linear(hidden_size[-1], (num_of_outputs+1)*num_of_outputs//2)
        # )
        self.p_sigma = nn.Sequential(
            nn.Linear(hidden_size[-1], num_of_outputs),
            nn.Sigmoid()
        )

    def forward(self, x, action=None, collision=None):

        x = super()._extract(x, action, collision)
        p_delta = self.p_delta(x)
        p_sigma = self.p_sigma(x)

        out = {'p_delta': p_delta, 'p_sigma': p_sigma}
        return out

class PoseGaussianUnerDecoder(PoseBasicDecoder):
    def __init__(self, in_channels, frame_size,
                 after_compression_flat_size=1024, 
                 p_dropout=0.2, use_dropout=False,
                 hidden_size=[256, 256], num_of_outputs=4, 
                 action_space=3, use_act_embedding=False, 
                 use_collision_embedding=False,
                 emb_layers=2,
                 embedding_size=8):
        super().__init__(in_channels, frame_size,
                        after_compression_flat_size, 
                        p_dropout, use_dropout,
                        hidden_size, num_of_outputs, 
                        action_space, use_act_embedding, 
                        use_collision_embedding,
                        emb_layers,
                        embedding_size)
       
        self.p_sigma = nn.Sequential(
            nn.Linear(hidden_size[-1], (num_of_outputs+1)*num_of_outputs//2)
        )

    def forward(self, x, action=None, collision=None):

        x = super()._extract(x, action, collision)
        p_delta = self.p_delta(x)
        p_sigma = self.p_sigma(x)
        
        tril_indices = torch.tril_indices(self.num_of_outputs, self.num_of_outputs)
        trils = torch.zeros(p_sigma.size(0),self.num_of_outputs, self.num_of_outputs)
        for i,(r,c) in enumerate(zip(tril_indices[0],tril_indices[1])):
            trils[:, r, c] = p_sigma[:, i]
        p_sigma = trils.matmul(torch.transpose(trils, -2, -1)).to(p_sigma.device)

        out = {'p_delta': p_delta, 'p_sigma': p_sigma}
        return out

class PoseNIGDecoder(PoseBasicDecoder):
    def __init__(self, 
                 in_channels, 
                 frame_size,
                 after_compression_flat_size=1024, 
                 p_dropout=0.2, 
                 use_dropout=False,
                 hidden_size=[256, 256], 
                 num_of_outputs=4, 
                 action_space=3, 
                 use_act_embedding=False, 
                 use_collision_embedding=False,
                 emb_layers=2,
                 embedding_size=8):
        super().__init__(in_channels, frame_size,
                        after_compression_flat_size, 
                        p_dropout, use_dropout,
                        hidden_size, num_of_outputs, 
                        action_space, use_act_embedding, 
                        use_collision_embedding,
                        emb_layers,
                        embedding_size)
        
        self.mu_de = nn.Linear(hidden_size[-1], num_of_outputs)
        self.logv_de = nn.Linear(hidden_size[-1], num_of_outputs)
        self.logalpha = nn.Linear(hidden_size[-1], num_of_outputs)
        self.logbeta = nn.Linear(hidden_size[-1], num_of_outputs)

        self.evidence = nn.Softplus()

    def forward(self, x, action=None, collision=None):

        x = super()._extract(x, action, collision)

        out = {}

        out['p_delta'] = self.mu_de(x)
        out['v'] = self.evidence(self.logv_de(x))
        out['alpha'] = self.evidence(self.logalpha(x)) + 1
        out['beta'] = self.evidence(self.logbeta(x))

        return out

if __name__ == '__main__':
    dum = torch.ones(2,512, 10, 6)
    pode = PoseBasicDecoder(512)
    de,si = pode(dum,  torch.ones(2).long())
    print(de.shape, si.shape)

    b = torch.randn(1,4,4)
    print(b)
    print(torch.transpose(b,-2,-1))

    a = torch.randn(10,10)
    tril_indices = torch.tril_indices(4, 4)
    print(tril_indices)
    print(tril_indices.shape)
    print(tril_indices.split(1))
    for r,c in zip(tril_indices[0],tril_indices[1]):
        print(r,c)
    trils = torch.zeros(10, 4, 4)
    for i, (r, c) in enumerate(zip(tril_indices[0], tril_indices[1])):
        trils[:, r, c] = a[:, i]
    p_sigma = trils.matmul(torch.transpose(trils, -2, -1))
    
    print(p_sigma.shape)
    a = torch.randn(1,3)
    a = a.matmul(torch.randn(3,1))
    print(a)

    a = torch.tensor([[ 0.2730,  0.0000,  0.0000,  0.0000],
             [-0.0123,  0.0459,  0.0000,  0.0000],
             [ 0.0903, -0.1800, -0.0789,  0.0000],
             [ 0.1957, -0.0635,  0.1443, -0.1727]])

    a = torch.tensor([[-0.1028,  0.0000,  0.0000,  0.0000],
             [-0.0501, -0.0390,  0.0000,  0.0000],
             [-0.0018, -0.2189, -0.0726,  0.0000],
             [-0.0399, -0.0500,  0.2153, -0.0185]])
    
    print(a.shape)
    b = a.matmul(a.T)
    print(torch.det(b))
    print(torch.linalg.det(b))
    print(b.det())

    import numpy as np
    # a = np.array([[ 3.9996e-02, -1.2133e-02, -2.1813e-03,  2.3166e-04],
    #          [-1.2133e-02,  1.3290e-01, -5.1497e-02,  1.6884e-01],
    #          [-2.1813e-03, -5.1497e-02,  2.6899e-02, -6.6576e-02],
    #          [ 2.3166e-04,  1.6884e-01, -6.6576e-02,  2.3183e-01]])
    
    a = np.array([[ 9.9597e-02, -8.3395e-02, -7.2180e-02, -2.7425e-02],
             [-8.3395e-02,  2.1709e-01,  6.2930e-02,  2.0362e-01],
             [-7.2180e-02,  6.2930e-02,  1.6713e-01, -6.6425e-02],
             [-2.7425e-02,  2.0362e-01, -6.6425e-02,  2.9877e-01]])
    
    
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)
    print(is_pos_def(a))
    print(np.log(np.linalg.det(a)))