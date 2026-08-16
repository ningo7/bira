"""Primitive BIRA weight layouts shared by model deployment adapters."""

import torch


def split_int16_byte_planes(tensor):
    """Return K0/K1 uint8 planes for a signed int16 tensor."""

    if tensor.dtype != torch.int16:
        raise ValueError('byte-plane splitting requires an int16 tensor')
    bits = tensor.to(torch.int32)
    low = (bits & 0xff).to(torch.uint8)
    high = ((bits >> 8) & 0xff).to(torch.uint8)
    return torch.stack((low, high), dim=0).contiguous()


def pack_binary_last_axis(tensor, channels=16):
    """Pack positive-valued binary channels with channel zero in the LSB."""

    if tensor.shape[-1] != channels:
        raise ValueError(
            f'binary axis has {tensor.shape[-1]} channels, expected {channels}'
        )
    bit_indices = torch.arange(
        channels, dtype=torch.int32, device=tensor.device
    )
    packed = torch.sum(
        (tensor > 0).to(torch.int32) << bit_indices,
        dim=-1,
    )
    if channels <= 16:
        return packed.to(torch.uint16).contiguous()
    if channels <= 32:
        return packed.to(torch.int32).contiguous()
    raise ValueError('binary packing supports at most 32 channels')
