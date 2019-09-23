import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import torch, torch.nn as nn
import torch.nn.functional as F

LEN_SEQ = 20



class EMBEDDING_LAYER(nn.Module):
    def __init__(self, matrix_for_embedding):
        super(self.__class__, self).__init__()

        """
        :param: matrix_for_embedding : word to vec matrix 
        
        """
        self.matrix = torch.tensor(matrix_for_embedding).float().cpu()

    def forward(self, input):
        output = torch.matmul(input, self.matrix)
        return (output)


class SMALL_V2V(nn.Module):
    def __init__(self, matrix_for_embedding, n_tokens, emb_size=150, lstm_units=50):
        super(self.__class__, self).__init__()
        """
        
        simplest LSTM module for sequence to sequence
        
        :input:  vectorized batch of ingridients
        
        :output: sequence with vocabulary size == n_tokens
        
        """
        self.lstm_units = lstm_units

        self.emb = nn.Sequential(EMBEDDING_LAYER(matrix_for_embedding),
                                 nn.Linear(300, emb_size)
                                 )

        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)

        self.logits = nn.Sequential(nn.Linear(lstm_units, 512),
                                    nn.ELU(), nn.Dropout(0.5),
                                    nn.Linear(512, n_tokens)
                                    )

    def forward(self, text_ix):
        initial_cell = torch.ones(text_ix.shape[0], self.lstm_units).cpu()
        initial_hid = torch.ones(text_ix.shape[0], self.lstm_units).cpu()

        captions_emb = self.emb(text_ix)

        lstm_out = self.lstm.forward(captions_emb, (initial_hid[None], initial_cell[None]))

        h = lstm_out[0]
        h1 = self.logits(h)
        h1 = h1 + 0.05 * h1 * torch.tanh(torch.randn_like(h1))
        return (h1)


class decoder(nn.Module):
    def __init__(self,matrix_for_embedding,  n_tokens, emb_size=100, out_size=30, lstm_units=60):
        super(self.__class__, self).__init__()
        """
        LSTM decoder class 
        
        :input:  vectorized batch of ingridients
        
        :output: vector : size == out_size //latent_space//
                
        
        """
        self.lstm_units = lstm_units

        self.emb = nn.Sequential(EMBEDDING_LAYER(matrix_for_embedding),
                                 nn.Linear(300, lstm_units * 4),
                                 nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(lstm_units * 4, emb_size))

        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)

        self.logits = nn.Sequential(nn.Linear(lstm_units * 2, 512),
                                    nn.ELU(), nn.Dropout(0.3), nn.BatchNorm1d(512),
                                    nn.Linear(512, out_size)
                                    )

    def forward(self, text_ix):
        initial_cell = torch.ones(text_ix.shape[0], self.lstm_units).cpu()
        initial_hid = torch.ones(text_ix.shape[0], self.lstm_units).cpu()

        captions_emb = self.emb(text_ix)

        lstm_out = self.lstm.forward(captions_emb, (initial_hid[None], initial_cell[None]))

        h = torch.cat([lstm_out[1][0][0], lstm_out[1][0][0]], dim=1)

        h1 = self.logits(h)

        return torch.sigmoid(h1)


class Generator(nn.Module):
    def __init__(self,matrix_for_embedding, LEN_SEQ,  n_tokens1, emb_size=100, lstm_units=100, rand_inp=50):
        super(self.__class__, self).__init__()

        """
        LSTM decoder class 

        :input:  vector : size == rand_inp

        :output: vectorized batch of ingridients


        """
        self.num_classes = n_tokens1

        self.dop = 0
        self.cnn_to_h0 = nn.Sequential(nn.Linear(rand_inp, lstm_units * 2),
                                       nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(lstm_units * 2),
                                       nn.Linear(lstm_units * 2, lstm_units))

        self.cnn_to_c0 = nn.Linear(rand_inp, lstm_units)

        self.cnn_to_h1 = nn.Sequential(nn.Linear(rand_inp, lstm_units * 2),
                                       nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(lstm_units * 2),
                                       nn.Linear(lstm_units * 2, lstm_units))

        self.cnn_to_c1 = nn.Linear(rand_inp, lstm_units)

        self.emb = nn.Sequential(EMBEDDING_LAYER(matrix_for_embedding),
                                 nn.Linear(300, lstm_units * 4),
                                 nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(lstm_units * 4, emb_size))

        self.lstm = nn.LSTM(emb_size, lstm_units, num_layers=2)

        self.logits1 = nn.Sequential(nn.Linear(lstm_units, lstm_units * 6),
                                     nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(lstm_units * 6, n_tokens1 * 2),
                                     nn.ELU(),
                                     nn.Linear(n_tokens1 * 2, n_tokens1))
    def one_hot_encoding_small(self, batch , batch_size):

        data = ((batch.reshape(batch_size, 1) == \
                 torch.arange(self.num_classes).reshape(1, self.num_classes)).reshape(batch_size, 1, self.num_classes)).float()
        return (data)
    def forward(self, noize, category, truth_caption):

        random_noize = torch.cat([noize, category], dim=-1)

        truth_caption_per = truth_caption.permute(1, 0)
        batch_size = random_noize.shape[0]

        captions_ix = torch.ones(1, batch_size).cpu().long()

        initial_hid = (torch.cat([self.cnn_to_h0(random_noize).unsqueeze(0), \
                                  self.cnn_to_h1(random_noize).unsqueeze(0)]))

        initial_cell = (torch.cat([self.cnn_to_c0(random_noize).unsqueeze(0), \
                                   self.cnn_to_c1(random_noize).unsqueeze(0)]))

        output1 = []

        for _idx in range(LEN_SEQ):

            enc = self.one_hot_encoding_small(captions_ix[-1] , batch_size)

            captions_emb = self.emb(enc.permute(1, 0, 2))

            lstm_out = self.lstm.forward(captions_emb, (initial_hid, initial_cell))

            logits1 = self.logits1(lstm_out[0])

            initial_hid, initial_cell = lstm_out[1]

            next_word_probs = logits1  # torch.sigmoid(logits1/5)

            if _idx <= self.dop:
                word = (truth_caption_per[_idx, :][None])

            else:
                word = next_word_probs.argmax(2)

            # print(word.shape)
            captions_ix = torch.cat([captions_ix, word[-1][None]])
            # print(captions_ix.shape)
            output1.append(logits1[-1][None])

        return {"ingridients": torch.softmax(torch.cat(output1).permute(1, 0, 2), dim=-1)}


class Classifier(nn.Module):
    def __init__(self, inp_size=300, out_size=20):
        super(self.__class__, self).__init__()
        self.solver = nn.Sequential(
            nn.Linear(inp_size, 1024),
            nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(512),
            nn.Linear(512, out_size)
        )

    def forward(self, inp):
        res = self.solver(inp)
        return torch.sigmoid(res)



class Generator_cor_map(nn.Module):
    def __init__(self, matrix_for_embedding, cor_tensor, LEN_SEQ, n_tokens1, emb_size=100, lstm_units=100, rand_inp=50):
        super(self.__class__, self).__init__()
        self.dop = 0

        self.num_classes = n_tokens1

        self.cor_tensor = cor_tensor
        self.cnn_to_h0 = nn.Sequential(nn.Linear(rand_inp, lstm_units * 2),
                                       nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(lstm_units * 2),
                                       nn.Linear(lstm_units * 2, lstm_units))

        self.cnn_to_c0 = nn.Linear(rand_inp, lstm_units)

        self.cnn_to_h1 = nn.Sequential(nn.Linear(rand_inp, lstm_units * 2),
                                       nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(lstm_units * 2),
                                       nn.Linear(lstm_units * 2, lstm_units))

        self.cnn_to_c1 = nn.Linear(rand_inp, lstm_units)

        self.emb = nn.Sequential(EMBEDDING_LAYER(matrix_for_embedding),
                                 nn.Linear(300, lstm_units * 4),
                                 nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(lstm_units * 4, emb_size))

        self.lstm = nn.LSTM(emb_size, lstm_units, num_layers=2)

        self.logits1 = nn.Sequential(nn.Linear(lstm_units, lstm_units * 6),
                                     nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(lstm_units * 6, n_tokens1 * 2),
                                     nn.ELU(),
                                     nn.Linear(n_tokens1 * 2, n_tokens1))

    def one_hot_encoding_small(self, batch , batch_size):

        data = ((batch.reshape(batch_size, 1) == \
                 torch.arange(self.num_classes).reshape(1, self.num_classes)).reshape(batch_size, 1, self.num_classes)).float()
        return (data)


    def forward(self, noize, category, truth_caption):

        random_noize = torch.cat([noize, category], dim=-1)

        truth_caption_per = truth_caption.permute(1, 0)
        batch_size = random_noize.shape[0]

        captions_ix = torch.ones(1, batch_size).cpu().long()

        initial_hid = (torch.cat([self.cnn_to_h0(random_noize).unsqueeze(0), \
                                  self.cnn_to_h1(random_noize).unsqueeze(0)]))

        initial_cell = (torch.cat([self.cnn_to_c0(random_noize).unsqueeze(0), \
                                   self.cnn_to_c1(random_noize).unsqueeze(0)]))

        output1 = []

        for _idx in range(LEN_SEQ):

            enc = self.one_hot_encoding_small(captions_ix[-1] , batch_size)

            captions_emb = self.emb(enc.permute(1, 0, 2))

            lstm_out = self.lstm.forward(captions_emb, (initial_hid, initial_cell))

            logits1 = torch.sigmoid(self.logits1(lstm_out[0]))

            initial_hid, initial_cell = lstm_out[1]
            # print(logits1.shape)

            if _idx > 3:
                next_word_probs = logits1 + 0.05 * logits1 * torch.tanh(torch.randn_like(logits1))

                next_word_probs = next_word_probs.double() + \
                                  _idx / 10 * logits1[0, 0].double() * self.cor_tensor[output1[-1].argmax(2)] + \
                                  _idx / 20 * logits1[0, 0].double() * self.cor_tensor[output1[-2].argmax(2)] + \
                                  _idx / 30 * logits1[0, 0].double() * self.cor_tensor[output1[-3].argmax(2)]
                next_word_probs = next_word_probs / (1 + _idx / 30 + _idx / 20 + _idx / 10)

                next_word_probs[0, 0, 0] = logits1[0, 0, 0] * 1.051
            else:
                next_word_probs = logits1 + 0.06 * logits1 * torch.tanh(torch.randn_like(logits1))
                next_word_probs[0, 0, 0] = logits1[0, 0, 0] * 1.051

            # plt.plot(next_word_probs[0][0].cpu().data.numpy())
            if _idx < self.dop:
                word = (truth_caption_per[_idx, :][None])

            else:
                word = next_word_probs.argmax(2)

            if word[0][0] in captions_ix and word[0][0] != 0:
                # print(_idx)
                # print(word[0][0])
                next_word_probs[:, :, word[0][0]] *= 0.3
                word = next_word_probs.argmax(2)

                if word[0][0] in captions_ix and word[0][0] != 0:
                    next_word_probs[:, :, word[0][0]] *= 0.3
                    word = next_word_probs.argmax(2)

            captions_ix = torch.cat([captions_ix, word[-1][None]])
            # print(logits1.shape)
            # print(next_word_probs.shape)
            output1.append(next_word_probs[-1][None].float())

        return {"ingridients": torch.softmax(torch.cat(output1).permute(1, 0, 2), dim=-1)}


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class EMBEDDING_LAYER_main(nn.Module):
    def __init__(self, matrix_for_embedding_main):
        super(self.__class__, self).__init__()

        # self.matrix = torch.ones(inp , out).cpu()
        self.matrix = torch.tensor(matrix_for_embedding_main).float().cpu()

    def forward(self, input):
        output = torch.matmul(input, self.matrix)
        return (output)


class text_generator2(nn.Module):
    def __init__(self, n_tokens1, n_tokens2, emb_size=300, \
                 lstm_units=500, emb_size2=200, lstm_units2=200):
        super(self.__class__, self).__init__()
        self.dop = 1
        self.lstm_units2 = lstm_units2
        self.emb2 = nn.Embedding(n_tokens2, emb_size2, padding_idx=0)
        self.lstm2 = nn.LSTM(emb_size2, lstm_units2, num_layers=1, batch_first=True)

        self.lstm_units = lstm_units
        self.emb = nn.Embedding(n_tokens1, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, lstm_units, num_layers=2, batch_first=True)

        rand_inp = lstm_units2 * 2
        self.cnn_to_h0 = nn.Sequential(nn.Linear(rand_inp, lstm_units * 2),
                                       nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(lstm_units * 2),
                                       nn.Linear(lstm_units * 2, lstm_units))

        self.cnn_to_c0 = nn.Linear(rand_inp, lstm_units)

        self.cnn_to_h1 = nn.Sequential(nn.Linear(rand_inp, lstm_units * 2),
                                       nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(lstm_units * 2),
                                       nn.Linear(lstm_units * 2, lstm_units))

        self.cnn_to_c1 = nn.Linear(rand_inp, lstm_units)

        self.logits = nn.Sequential(nn.Linear(lstm_units, 1512),
                                    nn.ELU(), nn.Dropout(0.3),
                                    nn.Linear(1512, n_tokens1)
                                    )

    def forward(self, ingridients):
        initial_cell_ = torch.ones(ingridients.shape[0], self.lstm_units2).cpu()
        initial_hid_ = torch.ones(ingridients.shape[0], self.lstm_units2).cpu()

        captions_emb2 = self.emb2(ingridients)

        lstm_out2 = self.lstm2.forward(captions_emb2, (initial_hid_[None], initial_cell_[None]))

        h = torch.cat([lstm_out2[1][0][0], lstm_out2[1][0][0]], dim=1)

        initial_hid = (torch.cat([self.cnn_to_h0(h).unsqueeze(0), \
                                  self.cnn_to_h1(h).unsqueeze(0)]))

        initial_cell = (torch.cat([self.cnn_to_c0(h).unsqueeze(0), \
                                   self.cnn_to_c1(h).unsqueeze(0)]))

        current_words = []

        captions_ix = torch.ones(ingridients.shape[0], 1).long().cpu()
        # captions_ix = text_ix[:,0:1]
        output = []
        for _idx in range(100):
            captions_emb = self.emb(captions_ix[:, -1:])
            lstm_out = self.lstm.forward(captions_emb, (initial_hid, initial_cell))

            initial_hid, initial_cell = lstm_out[1]
            logits1 = self.logits(lstm_out[0])
            logits1 = torch.sigmoid(logits1)

            logits1 = logits1  # + 0.002*logits1* torch.tanh(torch.randn_like(logits1))
            for w in [4, 19, 24, 31, 29, 247, 213, 32, 19, 390]:
                logits1[0][0][w] = 0

            word = logits1.argmax(2)

            app = int(word[0][0].cpu().data.numpy())

            if app in current_words[-12:] and _idx > 10:
                # logits1 = logits1 + 0.05*logits1* torch.tanh(torch.randn_like(logits1))
                logits1[0][0][app] = 0
                word = logits1.argmax(2)
                app = int(word[0][0].cpu().data.numpy())
                if app in current_words[-12:] and _idx > 10:
                    # logits1 = logits1 + 0.05*logits1* torch.tanh(torch.randn_like(logits1))
                    logits1[0][0][app] = 0
                    word = logits1.argmax(2)
                    app = int(word[0][0].cpu().data.numpy())
            # print(word)
            current_words.append(app)

            captions_ix = torch.cat([captions_ix, word], dim=1)

            output.append(logits1)

        return (torch.cat(output, dim=1))



def generate_text_direction(text_generation , array = [38,  72, 107,  74]  ):
    lenseq = len(array)
    my_vec = torch.tensor(array).cpu()
    my_vec_onehot = my_vec[None]
    #print(my_vec_onehot.shape)
    #text_generator.dop = lenseq
    result = text_generation(my_vec_onehot).argmax(2)
    #print(result.shape)
    return(result)