import os
from glob import glob

'''
    这个文件用于核查生成的数据是否产生错误
'''

name = ['train', 'val', 'val_mini']

def check_dataset():
    for n in name:
        print(f"------------------ {n} ------------------")
        output_dir = os.path.join('/home/yzc1/workspace/OccAnt/data/vo_dataset', n)
        scene_dir = sorted(glob(f"{output_dir}/*"))
        print("Total screens: ", len(scene_dir))

        a = []
        for i in scene_dir:
            if len(glob(f"{i}/depth/*")) != 4000:
                print("[Warning] dataset wrong: ", i)
                print("correct datasets len is 4000, but get: ", len(glob(f"{i}/rgb/*")))
            else:
                a.append(i.split('/')[-1])


        print('\t Loading all scenes from: /home/yzc1/workspace/OccAnt/data/datasets/pointnav/gibson/v2/' + n + '/content/*')

        all = [ i.split('/')[-1].split('.')[0] for i in glob('/home/yzc1/workspace/OccAnt/data/datasets/pointnav/gibson/v2/' + n + '/content/*')]
        print("\t All scenes: ", sorted(all))

        print(f"[Success] All is {len(all)} and {len(a)} scenes pass the check.")

        left = []
        for i in all:
            if i not in a:
                left.append(i)
        print(f"[Warning] dataset wrong nums: {len(left)}, they are {left}")
        print("\n")

if __name__ == "__main__":
    check_dataset()

