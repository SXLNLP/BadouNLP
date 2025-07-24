 def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        #随机正样本
        # if random.random() <= self.config["positive_sample_rate"]:
        #     p = random.choice(standard_question_index)
        #     #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        #     if len(self.knwb[p]) < 2:
        #         return self.random_train_sample()
        #     else:
        #         s1, s2 = random.sample(self.knwb[p], 2)
        #         return [s1, s2, torch.LongTensor([1])]
        # #随机负样本
        # else:
        #     p, n = random.sample(standard_question_index, 2)
        #     s1 = random.choice(self.knwb[p])
        #     s2 = random.choice(self.knwb[n])
        #     return [s1, s2, torch.LongTensor([-1])]

        p, n = random.choice(standard_question_index, 2)
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
            s3 = random.sample(self.knwb[n])
            return [s1,s2,s3]
