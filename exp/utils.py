import math
import torch

def concat_inputs(inputs, in_lens, factor=3, batch_first=False):
    if not batch_first:
        inputs = torch.transpose(inputs, 0, 1)
    batch_size, num_frames, feature_dims = inputs.shape
    remainder = num_frames % factor
    num_frames_to_pad = math.ceil(num_frames / factor) * factor - num_frames
    if num_frames_to_pad > 0:
        zero_tensor = torch.zeros(batch_size, num_frames_to_pad, feature_dims,
                                  dtype=inputs.dtype).to(inputs.device)
        inputs = torch.cat((inputs, zero_tensor), dim=1)
    inputs = inputs.reshape(
        batch_size, math.ceil(num_frames / factor), feature_dims * factor)
    in_lens = torch.ceil(in_lens / factor).long()
    if not batch_first:
        inputs = torch.transpose(inputs, 0, 1)
    return inputs, in_lens