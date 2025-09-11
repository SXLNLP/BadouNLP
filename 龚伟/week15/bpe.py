def bpe_encode(text, vocab_size=257, add_num=256):
    cover_num_list = lambda a: [byte for byte in a.encode('utf-8')]
    num_list = cover_num_list(text)

    merger_dict = {}
    for num_merger in range(vocab_size - add_num):
        max_char_num = get_max_char_num(num_list)
        bpe = merger_bpe(num_list, max_char_num, add_num)
        num_list = bpe
        merger_dict[max_char_num] = add_num
        add_num += 1
    return num_list, merger_dict


def bpe_decode(num_list, merger_dict):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merger_dict.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    tokens = b"".join(vocab[idx] for idx in num_list)
    text = tokens.decode("utf-8", errors="replace")
    return text


def get_max_char_num(num_list):
    char_num_dict = {}
    for pair in zip(num_list, num_list[1:]):
        char_num_dict[pair] = char_num_dict.get(pair, 0) + 1
    max_char_num = max(char_num_dict, key=char_num_dict.get)
    return max_char_num


def merger_bpe(num_list, top_p, add_num):
    new_num_list = []
    i = 0
    while i < len(num_list):
        if i < len(num_list) - 1 and num_list[i] == top_p[0] and num_list[i + 1] == top_p[1]:
            new_num_list.append(add_num)
            i += 2
        else:
            new_num_list.append(num_list[i])
            i += 1
    return new_num_list


def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        return text


def count_string(text):
    return len(set(text))


def main():
    # text = load_file("./corpus.txt")
    # len = count_string(text)

    text = "124213dsfnssfmsdfklnlSfn" * 1000
    len = count_string(text)
    num_list, merger_dict = bpe_encode(text, len)
    print(num_list, merger_dict)
    text = bpe_decode(num_list, merger_dict)
    print(text)


if __name__ == '__main__':
    main()