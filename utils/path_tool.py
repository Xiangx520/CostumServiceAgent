"""
为整个工程提供统一的绝对路径
"""
import os



def get_project_root() -> str:
    """
    获取工程所在的根目录
    :return: 字符串root path
    """
    
    # 当前文件的绝对路径
    curren_file = os.path.abspath(__file__)
    # 获取文件所在文件夹的绝对路径
    curren_dir = os.path.dirname(curren_file)
    # 获取工程根目录
    project_root = os.path.dirname(curren_dir)

    return project_root

def get_abs_path(relative_path: str) -> str:
    """
    从相对路径解析绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    root = get_project_root()
    return os.path.join(root, relative_path)




if __name__ == '__main__':
    print(get_abs_path("config\\config.txt"))