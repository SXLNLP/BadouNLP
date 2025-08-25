text = """ 英雄名：潮汐猎人
背景故事：人称利维坦的潮汐猎人曾经是沉没之岛的竞技大师，他的行为方式和他的同胞们一样神秘莫测。陆地居民都知道海上航线的重要性，帝国的崛起和衰亡都取决于谁掌控了公海。相比之下，海底航线和南海流民的敌对部族如何通过无数次小规模海底战斗开辟家园的故事，则鲜为人知。在南海人和陆地人那脆弱的协议中，我们可以瞥见这个海底王国的规模，不过他们的政体则显得十分复杂和不透明。利维坦似乎厌倦了这些琐碎的小冲突，便独自离开了，只效忠于他的深海之神-深渊触手麦尔朗恩。现在他在浅滩上独自游弋，寻找路上遇到的人类或者南海人，他对他的宿敌-舰队统帅昆卡有着一种独特的厌恶，然而他们为何成为敌人，已经和当年的激烈海战一起被遗忘了。
技能1：巨浪, 技能描述：召唤一股巨浪攻击一个敌方单位，减速并削弱护甲。
技能2：海妖外壳, 技能描述：加厚潮汐猎人的外皮，可以被动格挡物理伤害，当受到的伤害达到临界值时外皮还将移除绝大多数负面效果。

不与带有伤害格挡的物品叠加。

驱散类型：强驱散
技能3：锚击, 技能描述：潮汐猎人挥动他巨大的锚攻击附近的敌人，攻击时获得攻击力加成，同时降低敌人的攻击力。
技能4：深海触须, 技能描述：潮汐猎人释放出大量触手，对触及的敌人造成伤害和眩晕。触须造成%damage_pct%%%毁灭的伤害和眩晕，距离为%range_pct%%%毁灭的作用范围。
技能5：毁灭, 技能描述：猛击地面，触手向各个方向穿出，伤害并眩晕附近所有敌方单位。
"""

tokens = text.encode("utf-8")  # raw bytes
tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255 for convenience


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


vocab_size = 276  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens)  # copy so we don't destroy the original list

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

# decoding
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


print(decode(
    [260, 141, 260, 142, 229, 184, 166, 230, 156, 137, 228, 188, 164, 229, 174, 179, 230, 160, 188, 230, 140, 161, 258,
     231, 137, 169, 229, 147, 129, 229, 143, 160, 229, 138, 160, 268]))


# encoding
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


print(encode("不与带有伤害格挡的物品叠加。"))
