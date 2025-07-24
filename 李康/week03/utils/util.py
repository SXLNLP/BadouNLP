import numpy as np

def print_class_num(str, data):
    class_num = len(data[0])
    data_num = len(data)
    class_list = [0] * class_num
    for val in data:
        class_list[np.argmax(val)] += 1
    print(f"{str}:{class_list}")

def is_same_class(y_gt, y_p):
    return np.argmax(y_gt) == np.argmax(y_p)

if __name__ == '__main__':
    list = [[1,0], [0, 1], [1,0]]
    print_class_num("hello", np.array(list))
