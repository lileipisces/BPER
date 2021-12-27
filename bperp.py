from transformers import BertModel, BertTokenizer
import torch.nn as nn
import random
import torch
import math


class Batchify:
    def __init__(self, train_tuple_list, text_list, model_name, seq_max_len, batch_size, device):
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        # inherits from PreTrainedTokenizer (https://huggingface.co/transformers/preprocessing.html)
        encoded_inputs = tokenizer(text_list, add_special_tokens=True, padding='max_length', truncation=True, max_length=seq_max_len)
        self.input_ids = torch.tensor(encoded_inputs['input_ids']).to(device)
        self.attention_masks = torch.tensor(encoded_inputs['attention_mask']).to(device)
        self.train_tuple_list = train_tuple_list
        self.exp_list = list(range(len(text_list)))

        self.user2exp_set = {}
        self.item2exp_set = {}
        for x in self.train_tuple_list:
            u = x[0]
            i = x[1]
            exps = x[2]
            if u in self.user2exp_set:
                self.user2exp_set[u] |= exps
            else:
                self.user2exp_set[u] = exps.copy()
            if i in self.item2exp_set:
                self.item2exp_set[i] |= exps
            else:
                self.item2exp_set[i] = exps.copy()

        self.batch_size = batch_size
        self.sample_num = len(train_tuple_list)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.device = device

    def form_sample(self, u, i, exps):
        e = random.choice(list(exps))
        u_exp = self.user2exp_set[u]
        i_exp = self.item2exp_set[i]
        e_ = e
        while e_ in u_exp:
            e_ = random.choice(self.exp_list)
        e__ = e
        while e__ in i_exp:
            e__ = random.choice(self.exp_list)
        return e, e_, e__

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        user, item, exp, exp_, exp__ = [], [], [], [], []
        for idx in self.index_list[start:offset]:
            [u, i, exps] = self.train_tuple_list[idx]
            e, e_, e__ = self.form_sample(u, i, exps)
            user.append(u)
            item.append(i)
            exp.append(e)
            exp_.append(e_)
            exp__.append(e__)
        exp3 = exp + exp_ + exp__

        user_batch = torch.tensor(user, dtype=torch.int64).to(self.device)
        item_batch = torch.tensor(item, dtype=torch.int64).to(self.device)
        text_batch = self.input_ids[exp3]
        mask_batch = self.attention_masks[exp3]
        exp_batch = torch.tensor(exp3, dtype=torch.int64).to(self.device)
        return user_batch, item_batch, exp_batch, text_batch, mask_batch

    def prediction_batch(self, test_tuple_list):
        user, item = [], []
        for x in test_tuple_list:
            user.append(x[0])
            item.append(x[1])
        user_all = torch.tensor(user, dtype=torch.int64).to(self.device)
        item_all = torch.tensor(item, dtype=torch.int64).to(self.device)
        exp_all = torch.tensor(self.exp_list, dtype=torch.int64).to(self.device)
        return user_all, item_all, exp_all, self.input_ids, self.attention_masks


class BPERp(nn.Module):
    def __init__(self, batch_size, user_num, item_num, exp_num, model_name, hidden_size, dimension):
        super(BPERp, self).__init__()
        self.batch_size = batch_size
        self.user_embeddings = nn.Embedding(user_num, dimension)
        self.item_embeddings = nn.Embedding(item_num, dimension)
        self.exp_u_embeddings = nn.Embedding(exp_num, dimension)
        self.exp_i_embeddings = nn.Embedding(exp_num, dimension)
        self.exp_u_bias = nn.Embedding(exp_num, 1)
        self.exp_i_bias = nn.Embedding(exp_num, 1)
        self.bert = BertModel.from_pretrained(model_name)  # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        self.linear = nn.Linear(hidden_size, dimension)
        self.logsigmoid = nn.LogSigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.005
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.exp_u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.exp_i_embeddings.weight.data.uniform_(-initrange, initrange)
        self.exp_u_bias.weight.data.zero_()
        self.exp_i_bias.weight.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def get_all_text_embeddings(self, text, mask):
        sample_num = text.size(0)
        total_step = int(math.ceil(sample_num / self.batch_size))
        text_emb = []
        for step in range(total_step):
            start = step * self.batch_size
            end = start + self.batch_size
            text_batch = text[start:end]
            mask_batch = mask[start:end]
            hidden = self.bert(input_ids=text_batch, attention_mask=mask_batch)[0]  # (batch_size, max_len, hidden_size)
            t_emb = self.linear(hidden[:, 0, :])  # (batch_size, dimension)
            text_emb.append(t_emb)
        return torch.cat(text_emb, 0)  # (all_exp, dimension)

    def forward(self, user, item, exp, text, mask, top_k=0, mu=0):
        user_emb = self.user_embeddings(user)  # (batch_size, dimension) or (all_ui, dimension)
        item_emb = self.item_embeddings(item)

        if top_k == 0:
            # training
            hidden = self.bert(input_ids=text, attention_mask=mask)[0]  # (batch_size * 3, max_len, hidden_size)
            text_emb = self.linear(hidden[:, 0, :])  # (batch_size * 3, dimension)
            batch_size = user.size(0)
            eu_emb = self.exp_u_embeddings(exp[:batch_size]) * text_emb[:batch_size]  # (batch_size, dimension)
            eu_emb_ = self.exp_u_embeddings(exp[batch_size:(batch_size * 2)]) * text_emb[batch_size:(batch_size * 2)]
            ei_emb = self.exp_i_embeddings(exp[:batch_size]) * text_emb[:batch_size]
            ei_emb_ = self.exp_i_embeddings(exp[(batch_size * 2):]) * text_emb[(batch_size * 2):]
            eu_b = self.exp_u_bias(exp[:batch_size])  # (batch_size, 1)
            eu_b_ = self.exp_u_bias(exp[batch_size:(batch_size * 2)])
            ei_b = self.exp_i_bias(exp[:batch_size])
            ei_b_ = self.exp_i_bias(exp[(batch_size * 2):])

            u_side = user_emb * (eu_emb - eu_emb_) + (eu_b - eu_b_)  # (batch_size, dimension)
            i_side = item_emb * (ei_emb - ei_emb_) + (ei_b - ei_b_)
            score = self.logsigmoid(torch.sum(torch.cat([u_side, i_side], 0), 1))  # (batch_size * 2,)
            loss = -torch.mean(score)  # scalar
            return loss
        else:
            # prediction
            exp_u_emb = self.exp_u_embeddings(exp)  # (all_exp, dimension)
            exp_i_emb = self.exp_i_embeddings(exp)
            exp_u_b = self.exp_u_bias(exp)  # (all_exp, 1)
            exp_i_b = self.exp_i_bias(exp)
            t_emb = self.get_all_text_embeddings(text, mask)  # (all_exp, dimension), batchify to deal with memory issue

            eu_emb = (exp_u_emb * t_emb).t()  # (dimension, all_exp)
            ei_emb = (exp_i_emb * t_emb).t()
            eu_b = exp_u_b.t()  # (1, all_exp)
            ei_b = exp_i_b.t()

            sample_num = user.size(0)
            total_step = int(math.ceil(sample_num / self.batch_size))
            test_predict = []
            for step in range(total_step):
                start = step * self.batch_size
                end = start + self.batch_size
                u_batch = user_emb[start:end]  # (batch_size, dimension), batchify to deal with memory issue
                i_batch = item_emb[start:end]
                u_score = u_batch.matmul(eu_emb) + eu_b  # (batch_size, all_exp)
                i_score = i_batch.matmul(ei_emb) + ei_b
                score = u_score * mu + i_score * (1 - mu)
                topk = torch.topk(score, top_k, 1)[1]  # (batch_size, topk)
                test_predict.extend(topk.tolist())
            return test_predict  # list with size (all_ui, topk)
