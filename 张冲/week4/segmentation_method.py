WORD_DICT = {
    '北京',
    '大学',
    '北京大学',
    '大学生',
    '生前',
    '前来',
    '报到',
    '来',
    '生'
}


def segment(sequence, word_dict):
    res = []

    def dps_str(seq, start_index, seg_list: list):
        if start_index >= len(seq):
            res.append(seg_list.copy())
            return
        for i in range(start_index, len(seq)):
            seg = seq[start_index:i + 1]
            if seg in word_dict:
                seg_list.append(seg)
                dps_str(seq, i + 1, seg_list)
                seg_list.pop()

    dps_str(sequence, 0, [])
    return res


print(segment('北京大学生前来报到', WORD_DICT))
