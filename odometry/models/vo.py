import torch
import torch.nn as nn
from .resnet_encoder import ResnetEncoder
from .pose_decoder import *
import copy

class VoNet(nn.Module):
    def __init__(self,
                 num_layers,
                 frame_width,
                 frame_height,
                 decoder_type: str='base',
                 split_action: bool=False,
                 after_compression_flat_size=2048,
                 p_dropout=0.2,
                 use_dropout=False,
                 hidden_size=[256, 256],
                 num_input_images=2,
                 pose_type: str = 'SE2',
                 action_space=3,
                 emb_layers=2,
                 use_act_embedding=False,
                 use_collision_embedding=False,
                 embedding_size=8,
                 pretrained=False,
                 use_group_norm=False):
        super(VoNet, self).__init__()

        self.encoder = ResnetEncoder(num_layers=num_layers,
                                     pretrained=pretrained,
                                     num_input_images=num_input_images,
                                     use_group_norm=use_group_norm)
        
        if pose_type == 'SE2':
            num_of_outputs = 4
        elif pose_type == 'SE3':
            num_of_outputs = 6
        else:
            raise TypeError(f'{pose_type} is not exist, change type to [SE2|SE3]')
        
        if decoder_type == 'base':
            pose_decoder_type = PoseBasicDecoder
        elif decoder_type == 'laplacian':
            pose_decoder_type = PoseLaplacianUnerDecoder
        elif decoder_type == 'gaussian':
            pose_decoder_type = PoseGaussianUnerDecoder
        elif decoder_type == 'nig':
            pose_decoder_type = PoseNIGDecoder
        else:
            raise NotImplementedError

        frame_size = self._compute_encoder_output_size(frame_width, frame_height, num_input_images)
        pose_decoder = pose_decoder_type(in_channels=self.encoder.num_ch_enc[-1],
                                            frame_size=frame_size,
                                            p_dropout=p_dropout,
                                            after_compression_flat_size=after_compression_flat_size,
                                            hidden_size=hidden_size,
                                            use_dropout=use_dropout,
                                            use_act_embedding=use_act_embedding,
                                            use_collision_embedding=use_collision_embedding,
                                            embedding_size=embedding_size,
                                            action_space=action_space,
                                            emb_layers=emb_layers,
                                            num_of_outputs=num_of_outputs)
        self.action_map = {0: 'move_forward', 1: 'turn_left', 2: 'turn_righ'}

        self.split_action = split_action
        if split_action:
            assert decoder_type=='base', "Split action only support base decoder."
            self.pose_decoder = {self.action_map[i]: copy.deepcopy(pose_decoder) for i in range(action_space)} 
        else:
            self.pose_decoder = {'all': copy.deepcopy(pose_decoder)}


        self.pose_decoder = nn.ModuleDict(self.pose_decoder)

        # self.depth_de = DepthBasicDecoder(self.encoder.num_ch_enc)
        # self.depth_uncer_de = DepthBasicDecoder(self.encoder.num_ch_enc)

    def forward(self, x, action=None, collision=None, decotype='all'):

        x = self.encoder(x)

        out = self.pose_decoder[decotype](x, action, collision)

        return out

    def _compute_encoder_output_size(self, frame_width, frame_height, num_input_images):
        dummy_input = torch.randn(1, 3*num_input_images, frame_height, frame_width)
        out = self.encoder(dummy_input)[-1]

        return out.shape[-2:]

    

# net = VoNet(18, frame_width=320, frame_height=192, hidden_size=[256])
# torch.save(net.state_dict(),'a.pth')
# a = torch.rand(4,6,320,192)
# pose_delta, pose_uncer, depth, depth_uncer = net(a,torch.ones(4).long())
# print(pose_delta.shape)
# print(pose_uncer.shape)
# print(depth.shape)
# print(depth_uncer.shape)