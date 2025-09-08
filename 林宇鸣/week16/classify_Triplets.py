# 定义文件名
input_file = 'all_triplets_GEM.txt'
output_file_entity_attr = 'triplets_enti_attr_value_GEM.txt'
output_file_head_rel_tail = 'triplets_head_rel_tail_GEM.txt'

# 初始化两个列表来存储分类后的三元组
entity_attr_value_triplets = []
head_rel_tail_triplets = []

# 定义一个用于识别属性名的列表
attribute_names = [
    '中文名', '英文名', '出生日期', '出生地', '国籍', '职业', '音乐风格',
    '代表作品', '签约公司', '出道年份', '获奖情况', '粉丝称呼', '专辑销量',
    '歌曲时长', '发行时间', '所属专辑', '歌词主题', '新歌',
    '喜爱的食物', '喜爱的运动', '喜爱的饮料', '喜爱的颜色',
    '家庭背景', '生活理念', '生活方式', '公益活动参与', '文化交流活动'
]

# 添加与歌曲相关的属性
song_attribute_names = [
    '歌曲时长', '发行时间', '所属专辑', '歌词主题', '音乐风格'
]

# 合并属性名列表
all_attribute_names = attribute_names + song_attribute_names

# 读取输入文件
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除行末的换行符
        line = line.strip()
        # 用制表符分割三元组
        parts = line.split('\t')

        # 确保三元组有三个部分
        if len(parts) == 3:
            head, rel, tail = parts

            # 判断三元组类型
            if rel in all_attribute_names:
                entity_attr_value_triplets.append(line)  # 实体-属性名-属性值
            else:
                head_rel_tail_triplets.append(line)  # 实体-关系-实体

# 将实体-属性名-属性值写入文件
with open(output_file_entity_attr, 'w', encoding='utf-8') as file:
    for triplet in entity_attr_value_triplets:
        file.write(triplet + '\n')

# 将实体-关系-实体写入文件
with open(output_file_head_rel_tail, 'w', encoding='utf-8') as file:
    for triplet in head_rel_tail_triplets:
        file.write(triplet + '\n')

print("分类完成！")