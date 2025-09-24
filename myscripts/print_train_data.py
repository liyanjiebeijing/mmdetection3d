import pickle

def print_dict_tree(data, indent=0, prefix='', processed_keys=None):
    """
    递归打印字典的树形结构，处理键名前4字符相同的情况
    :param data: 要处理的数据(字典或列表)
    :param indent: 当前缩进级别
    :param prefix: 连接线前缀
    :param processed_keys: 已处理的键名前缀集合
    """
    if processed_keys is None:
        processed_keys = set()

    if isinstance(data, dict):
        keys = list(data.keys())
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            connector = '└── ' if is_last else '├── '

            # 获取当前键的前4个字符
            key_prefix = str(key)[:4]

            # 检查是否需要跳过递归
            skip_recursion = key_prefix in processed_keys
            if not skip_recursion:
                processed_keys.add(key_prefix)

            print(prefix + connector + str(key))

            if not skip_recursion:
                new_prefix = prefix + ('    ' if is_last else '│   ')
                print_dict_tree(data[key], indent + 1, new_prefix, processed_keys)

    elif isinstance(data, list) and len(data) > 0:
        id = len(data) // 2
        print(prefix + f'├── [{id}]')
        print_dict_tree(data[id], indent + 1, prefix + '│   ', processed_keys)


def main():
    with open('data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
        data = pickle.load(f)
        print_dict_tree(data)


if __name__ == "__main__":
    main()