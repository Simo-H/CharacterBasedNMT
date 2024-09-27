import torch
from torch.nn import Transformer
from .seq2seq import padding_mask
from .charactertokenizer import MASK_TOKEN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, optimizer, train_dataloader, loss_fn):
    """
    Train the model for one epoch
    :param model:
    :param optimizer:
    :param train_dataloader:
    :param loss_fn:
    :return: epoch loss
    """
    model.train()
    losses = 0

    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=DEVICE).type(torch.bool)
        src_padding_mask = padding_mask(src, MASK_TOKEN)
        tgt_padding_mask = padding_mask(tgt_input, MASK_TOKEN)
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt_input.size(0)).to(DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        # reduce lr on plateau

        # print("Batch: {0}, Loss: {1}".format(i, loss.detach().item()))
        losses += loss.detach().item()
        del src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, logits, tgt_input, tgt_out, loss
        torch.cuda.empty_cache()
    return losses / len(list(train_dataloader))


def evaluate(model, validation_dataloader, loss_fn):
    """
    Evaluate the model on a validation set
    :param model:
    :param validation_dataloader:
    :param loss_fn:
    :return:
    """
    model.eval()
    losses = 0

    for src, tgt in validation_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=DEVICE).type(torch.bool)
        src_padding_mask = padding_mask(src, MASK_TOKEN)
        tgt_padding_mask = padding_mask(tgt_input, MASK_TOKEN)
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt_input.size(0)).to(DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(validation_dataloader))