class Bpe_tokenizer:
    def __init__(self, vocab_size, training_text):
        self.train(vocab_size, training_text)

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
        newids = []
        i = 0
        while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, vocab_size, training_text):
        num_merges = vocab_size - 256
        tokens = training_text.encode("utf-8")
        ids = list(tokens)

        self.merges = {}  # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}, freq {stats[pair]}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            try:
                print(self.vocab[idx].decode("utf8"))
            except UnicodeDecodeError:
                continue
        print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

vocab_size = 512 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
training_text = """《黑神话：悟空》于2024年8月20日正式上线，迅速成为全网热议的焦点，其火爆程度令人瞩目。这款游戏之所以能在短时间内赢得如此高的关注，原因多方面且深入。\n\n首先，从文化内涵角度看，《黑神话：悟空》成功植根于中国深厚的历史文化土壤。《西游记》作为我国文学的经典之作，其主人公孙悟空已成为家喻户晓的英雄形象，承载着无数国人的童年记忆与深厚的文化情感。游戏以孙悟空为主角，让玩家亲身体验这位齐天大圣的神通广大与英勇无畏，这种文化认同感和情感共鸣成为《黑神话：悟空》火爆的重要基础。它不仅是一款游戏，更是一场文化的回归与盛宴，让玩家在游戏的世界里重新领略中国神话的魅力，以全新的、生动的方式展现传统文化。\n\n其次，在视觉呈现上，《黑神话：悟空》堪称一场视觉盛宴。制作团队不惜耗时耗力，运用先进的游戏制作技术，精心打造了美轮美奂的游戏画面。无论是细腻逼真的环境场景，还是栩栩如生的角色形象，再到炫目的技能特效，每一个细节都展现出了极高的制作水准。这种极致的视觉体验，极大地满足了玩家对游戏画面品质的追求，也是吸引众多玩家的关键因素之一。\n\n再者，游戏品质上，《黑神话：悟空》也达到了相当高的水平。游戏拥有丰富多样且极具挑战性的关卡设计，玩家需要运用智慧和技巧，不断探索、战斗，才能逐步推进游戏进程。角色的技能系统丰富且独特，玩家可以通过不同的技能组合，发挥出孙悟空的各种强大能力，增加了游戏的可玩性和策略性。同时，游戏的剧情紧凑且富有深度，在遵循原著故事框架的基础上，进行了大胆的创新和拓展，为玩家呈现了一个既熟悉又充满新鲜感的西游世界，让玩家在享受游戏乐趣的同时，也能感受到一个精彩绝伦的故事。\n\n最后，宣传推广策略也为《黑神话：悟空》的火爆添了一把柴。从2020年开始，制作方每年8月20日都会公开最新的实机视频，这些视频在网络上广泛传播，引发了大量关注和讨论，成功地为游戏上线预热造势。在社交媒体上，关于《黑神话：悟空》的话题热度持续攀升，玩家们纷纷自发地宣传分享，形成了强大的传播效应。此外，针对海外市场，制作方也积极开展宣传活动，通过号召海外网友参与视频投稿、与博主合作推广等方式，有效地扩大了游戏在国际上的影响力。\n\n《黑神话：悟空》的火爆并非偶然，而是其在文化内涵、视觉呈现、游戏品质以及宣传推广等多个方面共同发力的结果。它的成功，不仅为国产游戏树立了新的标杆，也证明了中国游戏产业在技术和创意上的巨大潜力。相信在《黑神话：悟空》的带动下，未来会有更多优秀的国产游戏涌现，推动中国游戏产业不断向前发展，让中国的游戏文化在全球舞台上绽放更加耀眼的光芒。同时，《黑神话：悟空》也为传统文化的传承与创新提供了新的思路和途径，让传统文化在现代社会中焕发出新的活力与生机。它不仅仅是一款游戏的成功，更是中国文化与现代科技融合发展的一个精彩范例，其影响力必将深远而持久。
2024年8月20日，国产游戏《黑神话：悟空》正式上线，迅速引发了全网的热议与追捧，其火爆程度令人惊叹。黑悟空之所以能如此之火，原因是多方面的。
从文化内涵来看，《黑神话：悟空》深深扎根于中国传统文化。《西游记》作为中国文学的经典之作，孙悟空更是家喻户晓的英雄形象，承载着无数国人的童年回忆和文化情感。该游戏以孙悟空为主角，让玩家能够在游戏中亲身扮演齐天大圣，体验其神通广大与英勇无畏，这种文化认同感和情感共鸣是黑悟空火爆的重要基础。它不仅仅是一款游戏，更像是一场文化的回归与盛宴，让玩家在游戏的世界里重新领略中国神话的魅力，使得传统文化以一种全新的、生动的方式呈现在大众面前。
在视觉呈现方面，黑悟空堪称一场视觉盛宴。制作团队不惜投入大量的时间和精力，运用先进的游戏制作技术，精心打造了美轮美奂的游戏画面。从细腻逼真的环境场景，到栩栩如生的角色形象，再到炫酷华丽的技能特效，每一个细节都展现出了极高的制作水准。无论是神秘奇幻的山林洞穴，还是气势恢宏的天庭宫殿，都仿佛让玩家身临其境，沉浸在一个充满想象力的神话世界之中。这种极致的视觉体验，极大地满足了玩家对于游戏画面品质的追求，也是吸引众多玩家的关键因素之一。
游戏品质上，黑悟空也达到了相当高的水平。它拥有丰富多样且极具挑战性的关卡设计，玩家需要运用智慧和技巧，不断探索、战斗，才能逐步推进游戏进程。角色的技能系统丰富且独特，玩家可以通过不同的技能组合，发挥出孙悟空的各种强大能力，增加了游戏的可玩性和策略性。同时，游戏的剧情紧凑且富有深度，在遵循原著故事框架的基础上，进行了大胆的创新和拓展，为玩家呈现了一个既熟悉又充满新鲜感的西游世界，让玩家在享受游戏乐趣的同时，也能感受到一个精彩绝伦的故事。
再者，宣传推广策略也为黑悟空的火爆添了一把柴。从 2020 年开始，制作方每年 8 月 20 日都会公开最新的实机视频，这些视频在网络上广泛传播，引发了大量关注和讨论，成功地为游戏上线预热造势。在社交媒体上，关于黑悟空的话题热度持续攀升，玩家们纷纷自发地宣传分享，形成了强大的传播效应。此外，针对海外市场，黑悟空也积极开展宣传活动，通过号召海外网友参与视频投稿、与博主合作推广等方式，有效地扩大了游戏在国际上的影响力。
《黑神话：悟空》的火爆并非偶然，而是其在文化内涵、视觉呈现、游戏品质以及宣传推广等多个方面共同发力的结果。它的成功，不仅为国产游戏树立了新的标杆，也证明了中国游戏产业在技术和创意上的巨大潜力。相信在黑悟空的带动下，未来会有更多优秀的国产游戏涌现，推动中国游戏产业不断向前发展，让中国的游戏文化在全球舞台上绽放更加耀眼的光芒。同时，黑悟空也为传统文化的传承与创新提供了新的思路和途径，让传统文化在现代社会中焕发出新的活力与生机。它不仅仅是一款游戏的成功，更是中国文化与现代科技融合发展的一个精彩范例，其影响力必将深远而持久。
"""
testing_text = "文章可读性较高，段落结构清晰，观点明确。为提高传播效果，建议简化句子结构，突出关键信息，如“《黑神话：悟空》融合传统文化与现代技术，成为国产游戏新标杆，推动产业创新发展。"

bpe_tokenizer = Bpe_tokenizer(vocab_size, training_text)
print(f"merges: {bpe_tokenizer.merges}")
print(f"vocab: {bpe_tokenizer.vocab}")
ids = bpe_tokenizer.encode(testing_text)
print(f"encoded tokens: {ids}")
text = bpe_tokenizer.decode(ids)
print(f"decoded text: {text}")
