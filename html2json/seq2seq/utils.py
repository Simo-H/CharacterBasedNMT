import torch
from torch.nn import Transformer
from html2json.charactertokenizer import CLS_TOKEN, SEP_TOKEN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def padding_mask(seq: torch.Tensor, pad_token):
    """
    Create a mask for the padding tokens
    :param seq:
    :param pad_token:
    :return:
    """
    return (seq == pad_token).transpose(0, 1)


def greedy_decode(model, src, src_mask, max_len):
    """
    Greedy decoding - generate the output sequence token by token and taking
    the token with the highest probability in each step
    :param model:
    :param src: src tensor
    :param src_mask: src mask
    :param max_len:
    :return: generated sequence
    """
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(CLS_TOKEN).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (Transformer.generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == SEP_TOKEN:
            break
    return ys


def translate(model: torch.nn.Module, src_sentence: str, html_tokenizer, json_tokenizer):
    """
    Translate a source sentence to a target sentence
    :param model:
    :param src_sentence:
    :param html_tokenizer:
    :param json_tokenizer:
    :return:
    """
    model.eval()
    src = torch.LongTensor(html_tokenizer.encode(src_sentence)).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 100).flatten()
    return "".join(json_tokenizer.decode(tgt_tokens.cpu().numpy()))
