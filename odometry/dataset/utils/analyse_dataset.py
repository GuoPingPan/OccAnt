import gzip
from glob import glob
import os.path as osp
from tqdm import tqdm
import json

'''
    这个文件用于分析数据集的总数
    结果：
        - train 数据集：共 72 个场景，每个场景共 50000 个 episodes
        - val 数据集：共 14 个场景，每个场景共 71 个 episodes
        - val_mini 数据集：共 3 个场景，每个场景共 71 个 episodes
'''

# import zipfile
#
# with zipfile.ZipFile(r'E:\workspace\ANM\pointnav_gibson_v2.zip') as file:
#     print(data)

train_datasets = glob(r'E:\workspace\ANM\pointnav_gibson_v2\train\content\*')
val_datasets = glob(r'E:\workspace\ANM\pointnav_gibson_v2\val\content\*')
val_mini_datasets = glob(r'E:\workspace\ANM\pointnav_gibson_v2\val_mini\content\*')


assert train_datasets is not None
assert val_datasets is not  None
assert val_mini_datasets is not None

def read_json_gz(datasets):
    scenes_name = {}
    for i in tqdm(datasets):
        scenes_name[i.split('\\')[-1].split('.')[0]] = 0
        with gzip.open(i) as dataset:
            data = dataset.readlines()
            data = str(data[0]).split('episode_id')
            scenes_name[i.split('\\')[-1].split('.')[0]] = len(data) - 1
    print(scenes_name)

read_json_gz(train_datasets)
read_json_gz(val_datasets)
read_json_gz(val_mini_datasets)


'''
输出结果：
{'Adrian': 50000, 'Albertville': 50000, 'Anaheim': 50000, 'Andover': 50000, 'Angiola': 50000, 'Annawan': 50000, 'Applewold': 50000, 'Arkansaw': 50000, 'Avonia': 50000, 'Azusa': 50000, 'Ballou': 49999, 'Beach': 50000, 'Bolton': 50000, 'Bowlus': 50000, 'Brevort': 50000, 'Capistrano': 50000, 'Colebrook': 50000, 'Convoy': 50000, 'Cooperstown': 50000, 'Crandon': 50000, 'Delton': 50000, 'Dryville': 50000, 'Dunmor': 50000, 'Eagerville': 50000, 'Goffs': 50000, 'Hainesburg': 50000, 'Hambleton': 50000, 'Haxtun': 50000, 'Hillsdale': 50000, 'Hometown': 50000, 'Hominy': 50000, 'Kerrtown': 49999, 'Maryhill': 50000, 'Mesic': 50000, 'Micanopy': 50000, 'Mifflintown': 50000, 'Mobridge': 50000, 'Monson': 50000, 'Mosinee': 50000, 'Nemacolin': 50000, 'Nicut': 50000, 'Nimmons': 50000, 'Nuevo': 50000, 'Oyens': 50000, 'Parole': 50000, 'Pettigrew': 50000, 'Placida': 50000, 'Pleasant': 50000, 'Quantico': 50000, 'Rancocas': 50000, 'Reyno': 50000, 'Roane': 50000, 'Roeville': 49999, 'Rosser': 50000, 'Roxboro': 50000, 'Sanctuary': 50000, 'Sasakwa': 50000, 'Sawpit': 49999, 'Seward': 50000, 'Shelbiana': 50000, 'Silas': 50000, 'Sodaville': 50000, 'Soldier': 50000, 'Spencerville': 50000, 'Spotswood': 50000, 'Springhill': 50000, 'Stanleyville': 50000, 'Stilwell': 50000, 'Stokes': 50000, 'Sumas': 50000, 'Superior': 50000, 'Woonsocket': 50000}
{'Cantwell': 71, 'Denmark': 71, 'Eastville': 71, 'Edgemere': 71, 'Elmira': 71, 'Eudora': 71, 'Greigsville': 71, 'Mosquito': 71, 'Pablo': 71, 'Ribera': 71, 'Sands': 71, 'Scioto': 71, 'Sisters': 71, 'Swormville': 71}
{'Greigsville': 71, 'Mosquito': 71, 'Pablo': 71}
'''