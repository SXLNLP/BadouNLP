def get_second_smallest(col1, col2, col3):
    """
    对比三列数据，排序后取倒数第二小的值

    参数:
    col1, col2, col3: 三个要比较的值，可以是数字或字符串

    返回:
    排序后的倒数第二个值
    """
    # 将三个值放入列表
    values = [col1, col2, col3]

    try:
        # 尝试转换为数字比较
        numeric_values = [
            float(str(v).strip(' \'"')) if str(v).replace('.', '', 1).replace('-', '', 1).isdigit() else float('inf')
            for v in values]

        # 如果有至少两个是数字，则按数字排序
        if sum(1 for v in numeric_values if v != float('inf')) >= 2:
            sorted_values = sorted(numeric_values)
            # 过滤掉无穷大值
            filtered = [v for v in sorted_values if v != float('inf')]
            return filtered[-2] if len(filtered) >= 2 else sorted_values[1]
        elif sum(1 for v in numeric_values if v != float('inf')) == 1:
            sorted_values = sorted(numeric_values)
            # 过滤掉无穷大值
            filtered = [v for v in sorted_values if v != float('inf')]
            return filtered[0]
        else:
            pass
    except:
        pass

    # 如果无法按数字比较，则按字符串比较
    str_values = [str(v) for v in values]
    sorted_str = sorted(str_values)
    return sorted_str[-2]
