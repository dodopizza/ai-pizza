import numpy as np
import torch

def vectorizer(sample, ingr_to_num, quant_to_num, how_to_num, LEN_SEQ=20):

    """
    :param sample:  recipe[ingridients] array
    :param LEN_SEQ:
    :param ingr_to_num:  tokenizer map for ingridients
    :param quant_to_num: tokenizer map for quantyty
    :param how_to_num:   tokenizer map for ingredient preparing configuration

    :return:  array with tokenizing [ingridients, quantyty, ingredient configuration]

    """
    ingridients = np.zeros(LEN_SEQ)
    quantity = np.zeros(LEN_SEQ)
    how_coook = np.zeros(LEN_SEQ)

    for i, word in enumerate(sample):
        if i < (LEN_SEQ):
            ingridients[i] = ingr_to_num[word[2]]
            try:
                quantity[i] = quant_to_num[word[0]]
                how_coook[i] = how_to_num[word[3]]
            except:
                #NULL paddings
                quantity[i] = 2
                how_coook[i] = 2

    return ([ingridients, quantity, how_coook])



def one_hot_encoding(batch , num_classes , batch_size, LEN_SEQ):
    data = []
    for idx in range(LEN_SEQ):
        data.append((batch[:,idx].reshape(batch_size,1) == \
                     torch.arange(num_classes).reshape(1, num_classes)).reshape(batch_size,1,num_classes))
    one_hot_batch = torch.cat(data, dim = 1).float()
    return(one_hot_batch)

def one_hot_encoding_fix(batch , num_classes , batch_size, lenseq):
    data = []
    for idx in range(lenseq):
        data.append((batch[:,idx].reshape(batch_size,1) == \
                     torch.arange(num_classes).reshape(1, num_classes)).reshape(batch_size,1,num_classes))
    one_hot_batch = torch.cat(data, dim = 1).float()
    return(one_hot_batch)


def one_hot_encoding_small(batch , num_classes , batch_size):

    data = ((batch.reshape(batch_size,1) == \
                     torch.arange(num_classes).reshape(1, num_classes)).reshape(batch_size,1,num_classes)).float()
    return(data)


def sample_data_batch(batch_size, data , n_tokens_ingr , n_tokens_quant , n_tokens_how , LEN_SEQ):
    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    index_y = 0
    igg = 0
    while True:

        idxs_ = idxs[index_y: index_y + batch_size]
        big_batch = torch.tensor(data[idxs_]).long()

        output = dict()
        output["ingridients"] = big_batch[:, 0].cpu()
        output["quantity"] = big_batch[:, 1].cpu()
        output["how_coook"] = big_batch[:, 2].cpu()

        ingridients_one_hot = one_hot_encoding(output["ingridients"], n_tokens_ingr, batch_size, LEN_SEQ)
        quantity_one_hot = one_hot_encoding(output["quantity"], n_tokens_quant, batch_size, LEN_SEQ)
        how_coook_one_hot = one_hot_encoding(output["how_coook"], n_tokens_how, batch_size, LEN_SEQ)

        output["ingridients_one_hot"] = ingridients_one_hot.cpu()
        output["quantity_one_hot"] = quantity_one_hot.cpu()
        output["how_coook_one_hot"] = how_coook_one_hot.cpu()

        index_y += batch_size

        if index_y + batch_size > data.shape[0]:
            break
        if igg > 49:
            break
        igg += 1
        yield output


def sample_data_batch_not_shuffle(batch_size, data , n_tokens_ingr , n_tokens_quant , n_tokens_how , LEN_SEQ):
    idxs = np.arange(data.shape[0])
    index_y = 0
    while True:

        idxs_ = idxs[index_y: index_y + batch_size]
        big_batch = torch.tensor(data[idxs_]).long()

        output = dict()
        output["ingridients"] = big_batch[:, 0].cpu()
        output["quantity"] = big_batch[:, 1].cpu()
        output["how_coook"] = big_batch[:, 2].cpu()

        ingridients_one_hot = one_hot_encoding(output["ingridients"], n_tokens_ingr, batch_size, LEN_SEQ)
        quantity_one_hot = one_hot_encoding(output["quantity"], n_tokens_quant, batch_size, LEN_SEQ)
        how_coook_one_hot = one_hot_encoding(output["how_coook"], n_tokens_how, batch_size, LEN_SEQ)

        output["ingridients_one_hot"] = ingridients_one_hot.cpu()
        output["quantity_one_hot"] = quantity_one_hot.cpu()
        output["how_coook_one_hot"] = how_coook_one_hot.cpu()

        index_y += batch_size

        if index_y + batch_size > data.shape[0]:
            break
        yield output