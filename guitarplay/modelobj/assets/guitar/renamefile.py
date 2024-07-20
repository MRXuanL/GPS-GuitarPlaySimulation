import os

def read_filenames(folder_path):
    # 获取文件夹中所有文件的文件名列表
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.obj')]
    filenames.sort()
    names=[]
    print("文件名列表:")
    for filename in filenames:
        names.append(filename[:-4])
        print(f'<mesh name="{filename[:-4]}" file="guitar/{filename}"/>')
    print("geom:")
    for i,filename in enumerate(filenames):
        print(f'<geom type="mesh" mesh="{names[i]}" size="1"/>')

    
# 指定文件夹路径
folder_path = "assets/guitar"

# 调用函数读取文件名
read_filenames(folder_path)