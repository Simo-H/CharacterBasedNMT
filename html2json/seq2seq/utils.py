import torch
from torch.nn import Transformer
from html2json.charactertokenizer import CLS_TOKEN, SEP_TOKEN
import torch.nn.functional as F
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


def translate_greedy_search(model: torch.nn.Module, src_sentence: str, html_tokenizer, json_tokenizer):
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


def beam_search(model, src, src_mask, start_token, end_token, beam_width=3, max_length=20):
    # Encode the source sequence
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    # Initialize the beam with the start token
    beams = [(torch.tensor([start_token]), 0)]  # List of tuples (sequence, score)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in beams:
            tgt = seq.to(DEVICE)  # Add batch dimension
            tgt_mask = (Transformer.generate_square_subsequent_mask(tgt.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(tgt, memory, tgt_mask)

            # Decode the current sequence
            # out = model.decode(tgt, memory)  # Shape (1, seq_len, d_model)
            logits = out[:, -1, :]  # Shape (1, vocab_size)
            probs = F.log_softmax(logits, dim=-1).squeeze(0)  # Shape (vocab_size)

            # Expand each beam with all possible next tokens
            for i in range(probs.size(-1)):
                candidate = (torch.cat([seq, torch.tensor([i])]), score + probs[i].item())
                all_candidates.append(candidate)

        # Select the top beam_width candidates
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

        # Check if all beams have reached the end token
        if all(seq[-1] == end_token for seq, _ in beams):
            break

    # Select the best beam
    best_sequence = sorted(beams, key=lambda x: x[1], reverse=True)[0][0]
    return best_sequence


def translate_beam_search(model: torch.nn.Module, src_sentence: str, html_tokenizer, json_tokenizer):
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
    tgt_tokens = beam_search(
        model,  src, src_mask, CLS_TOKEN, SEP_TOKEN, beam_width=1, max_length=1000).flatten()
    return "".join(json_tokenizer.decode(tgt_tokens.cpu().numpy()))
