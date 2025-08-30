from collections import defaultdict

corpus = ["我爱自然语言处理", "我爱机器学习", "自然语言处理很有趣"]

def initialize_corpus(corpus):
    tokenized_corpus = []
    for sentence in corpus:
        tokens = list(sentence)+["<eos>"]   
        tokenized_corpus.append(tokens)
    return tokenized_corpus

tokenized_corpus = initialize_corpus(corpus)
print("Tokenized Corpus:", tokenized_corpus)

def get_pair_frequencies(tokenized_corpus):
    pair_freq = defaultdict(int)
    for sentence in tokenized_corpus:
        for i in range(len(sentence)-1):
            pair = (sentence[i], sentence[i+1])
            pair_freq[pair] += 1
    return pair_freq

pair_freq = get_pair_frequencies(tokenized_corpus)
print("Pair Frequencies:", dict(pair_freq))

def merge_pairs(tokenized_corpus, pair_to_merge):
    new_corpus = []
    for sentence in tokenized_corpus:
        new_sentence = []
        i = 0
        while i < len(sentence):
            if i < len(sentence) - 1 and (sentence[i], sentence[i+1]) == pair_to_merge:
                new_sentence.append(sentence[i]+sentence[i+1])
                i += 2
            else:
                new_sentence.append(sentence[i])
                i += 1
        new_corpus.append(new_sentence)
    return new_corpus

pair_freq = get_pair_frequencies(tokenized_corpus)
pair_to_merge = max(pair_freq, key=pair_freq.get)
print("Pair to Merge:", pair_to_merge)

tokenized_corpus = merge_pairs(tokenized_corpus, pair_to_merge)
print("Updated Tokenized Corpus:", tokenized_corpus)

def train_bpe(corpus, num_merges):
    tokenized_corpus = initialize_corpus(corpus)
    vocab = set()
    for phrase in corpus:
        for char in phrase:
            vocab.add(char)
    vocab.add("<eos>")

    for _ in range(num_merges):
        pair_freq = get_pair_frequencies(tokenized_corpus)
        if not pair_freq:
            break
        pair_to_merge = max(pair_freq, key=pair_freq.get)
        tokenized_corpus = merge_pairs(tokenized_corpus, pair_to_merge)
        vocab.add(pair_to_merge[0]+pair_to_merge[1])    
        print(f"Merge: {pair_to_merge}, New Vocab Size: {len(vocab)}")
    return tokenized_corpus, vocab

num_merges = 5
final_corpus, final_vocab = train_bpe(corpus, num_merges)
print("Final Tokenized Corpus:", final_corpus)  