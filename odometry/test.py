import torch
from models.resnet_encoder import ResnetEncoder
from models.vo import VoNet

# model = VoNet(num_layers=18, frame_height=128, frame_width=128, split_action=True)
# # print(model.state_dict())

# for k,v in model.state_dict().items():
#     print(k)



# # model = ResnetEncoder(num_layers=18)

# print(type(model.state_dict()))
# # print(model.state_dict())
# import logging
# # 声明全局logger对象
# logger = logging.getLogger("")

# def init_logger():
#     # 配置logger对象
#     logger.setLevel(logging.DEBUG)
#     handler = logging.FileHandler('out.log')
#     # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # handler.setFormatter(formatter)
#     logger.addHandler(handler)

# def add_message():
#     logger.info("fdsaf")


# init_logger()
# add_message()
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for i in range(10):
    writer.add_scalars('Train', {'MoveFoward': i}, i)
    writer.add_scalars('Train', {'TrunLeft': i}, i)
