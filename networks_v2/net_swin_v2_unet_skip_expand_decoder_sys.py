import collections
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Tuple, Union, List, Set
from torch import Tensor
from einops import rearrange

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def meshgrid(
    *tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: Optional[str] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    # if is_torch_greater_or_equal_than_1_10:
    #     return torch.meshgrid(*tensors, indexing=indexing)
    # else:
    #     if indexing != "ij":
    #         raise ValueError('torch.meshgrid only supports `indexing="ij"` for torch<1.10.')
    #     return torch.meshgrid(*tensors)
    return torch.meshgrid(*tensors, indexing=indexing)


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->Swinv2
class Swinv2SelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Swinv2Attention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        self.self = Swinv2SelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )

        # don't know the meaning of self.output?
        self.output = Swinv2SelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->Swinv2
class Swinv2Intermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            # directly create gelu activation function
            self.intermediate_act_fn = GELUActivation()
            # self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput with Swin->Swinv2
class Swinv2Output(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->Swinv2
class Swinv2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Swinv2SelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=[0, 0]):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        self.pretrained_window_size = pretrained_window_size
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = (
            torch.stack(meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # [1, 2*window_height - 1, 2*window_width - 1, 2]
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        )
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # cosine attention
        attention_scores = F.normalize(query_layer, dim=-1) @ F.normalize(key_layer, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        # [window_height*window_width,window_height*window_width,num_attention_heads]
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        # [num_attention_heads,window_height*window_width,window_height*window_width]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# no need to modify (a complete swin transformer layer based on window/shifted partition)
class Swinv2Layer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, pretrained_window_size=0):
        super().__init__()
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.set_shift_and_window_size(input_resolution)
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swinv2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def set_shift_and_window_size(self, input_resolution):
        target_window_size = (
            self.window_size
            if isinstance(self.window_size, collections.abc.Iterable)
            else (self.window_size, self.window_size)
        )
        target_shift_size = (
            self.shift_size
            if isinstance(self.shift_size, collections.abc.Iterable)
            else (self.shift_size, self.shift_size)
        )
        self.window_size = (
            input_resolution[0] if input_resolution[0] <= target_window_size[0] else target_window_size[0]
        )
        self.shift_size = (
            0
            if input_resolution
            <= (
                self.window_size
                if isinstance(self.window_size, collections.abc.Iterable)
                else (self.window_size, self.window_size)
            )
            else target_shift_size[0]
        )

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_shift_and_window_size(input_dimensions)
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        # pad hidden_states to multiples of window size
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.layernorm_before(attention_windows)
        hidden_states = shortcut + self.drop_path(hidden_states) # hidden_states (B, N, C)

        # what does the following lines mean_teacher?
        # (intermediate + output) jointly constitute two layer MLP in transformer layer
        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


class Swinv2PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution

        # so far not sure about the order of expand and norm operations
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class Swinv2PatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # [batch_size, height/2 * width/2, 4*num_channels]
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # [batch_size, height/2 * width/2, 4*C]

        # the difference between this v2 and swin-t v1 is the order of reduction and norm operations
        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature)

        return input_feature


class Swinv2Stage(nn.Module):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample, pretrained_window_size=0
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                Swinv2Layer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    # Copied from transformers.models.swin.modeling_swin.SwinStage.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(layer_outputs[0], input_dimensions)
            # hidden_states = self.downsample(layer_outputs[0])
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class Swinv2Stage_up(nn.Module):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample, pretrained_window_size=0
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                Swinv2Layer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    # Copied from transformers.models.swin.modeling_swin.SwinStage.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        to_concat_feature: torch.Tensor = None # because need to integrate features from encoder
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

        if self.downsample is not None:
            # need to modify because this is patch expand
            # height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            height_downsampled, width_downsampled = int(height * 2), int(width * 2)
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            # hidden_states = self.downsample(layer_outputs[0], input_dimensions)
            hidden_states = self.downsample(layer_outputs[0])
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class Swinv2Decoder(nn.Module):
    def __init__(self, config, norm_layer=nn.LayerNorm):
        super(Swinv2Decoder, self).__init__()
        self.num_layers = len(config.depths_decoder)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.patches_resolution = self.config.patches_resolution
        self.norm_layer = norm_layer
        
        # print('patches_resolution is :', self.patches_resolution)
        # raise Exception

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(self.config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(self.config.embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                self.layers_up.append(
                    Swinv2PatchExpand(
                        input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(self.config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        dim_scale=2,
                        norm_layer=self.norm_layer)
                )
            else:
                self.layers_up.append(
                    Swinv2Stage_up(
                        config=self.config,
                        dim=int(self.config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        depth=self.config.depths[(self.num_layers - 1 - i_layer)],
                        num_heads=self.config.num_heads[(self.num_layers - 1 - i_layer)],
                        drop_path=dpr[sum(self.config.depths[:(self.num_layers - 1 - i_layer)]):sum(
                            self.config.depths[:(self.num_layers - 1 - i_layer) + 1])],
                        downsample=Swinv2PatchExpand if i_layer < self.num_layers-1 else None,
                        pretrained_window_size=pretrained_window_sizes[i_layer] # all same values
                    )
                )

            self.concat_back_dim.append(concat_linear)
            # self.layers_up.append(self.layers_up)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # input_dimensions: Tuple[int, int], # (H/16, W/16) or (patch_resolution/4, patch_resolution/4)
        concat_feature_list,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        # need to change to 'True' to get the output of each layer
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        concat_features = concat_feature_list
        for i, layer_module in enumerate(self.layers_up):
            if i == 0:
                # first
                hidden_states = layer_module(hidden_states)
                B, N, C = hidden_states.shape
                # compute input_dimensions
                input_dimensions = (int(math.sqrt(N)), int(math.sqrt(N)))

            else:
                """
                Should be this:
                
                hidden_states: (B, 484, 384) (B, W/16 * H/16, 4C)
                
                concat features[0]: (B, 7744, 96) (B, W/4 * H/4, C)
                concat features[1]: (B, 1936, 192) (B, W/8 * H/8, 2C)
                concat features[2]: (B, 484, 384) (B, W/16 * H/16, 4C)
                """
                
                # print(hidden_states.size())
                # raise Exception
                
                hidden_states = torch.cat([hidden_states, concat_features[3-i]], dim=-1)
                hidden_states = self.concat_back_dim[i](hidden_states)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)
                # only hidden_states is what we want (the output)
                hidden_states = layer_outputs[0]
                output_dimensions = layer_outputs[1]

                # compute input_dimensions
                input_dimensions = (output_dimensions[-2], output_dimensions[-1])

        return hidden_states

    # def forward_up_features(self, x, x_downsample):
    #
    #     return_layer_list = []
    #
    #     for inx, layer_up in enumerate(self.layers_up):
    #         if inx == 0:
    #             x = layer_up(x)
    #         else:
    #             x = torch.cat([x, x_downsample[3 - inx]], -1)
    #             x = self.concat_back_dim[inx](x)
    #             x = layer_up(x)
    #
    #         return_layer_list.append(x)
    #
    #     x = self.norm_up(x)  # B L C
    #
    #     return x, return_layer_list


class Swinv2Encoder(nn.Module):
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = nn.ModuleList(
            [
                Swinv2Stage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=Swinv2PatchMerging if (i_layer < self.num_layers - 1) else None,
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    # Copied from transformers.models.swin.modeling_swin.SwinEncoder.forward with SwinEncoderOutput->Swinv2EncoderOutput
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        # need to change to 'True' to get the output of each layer
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ):
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states, input_dimensions, layer_head_mask
                )
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            # only hidden_states is what we want (the output)
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)

            if output_hidden_states:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # not necessary to output Swinv2EncoderOutput class
        # return Swinv2EncoderOutput(
        #     last_hidden_state=hidden_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     reshaped_hidden_states=all_reshaped_hidden_states,
        # )

        # tuple (use all_hidden_states to implement the long-skip connection)
        # all_hidden_states ((B, (W/8*H/8), 2C),(B, (W/16*H/16), 4C),(B, (W/32*H/32), 8C),(B, (W/32*H/32), 8C)):
        # 4 outputs from 4 consecutive Swin transformer blocks
        
        return (hidden_states, all_hidden_states)


class Swinv2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


class Swinv2Embeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = Swinv2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, config, add_pooling_layer=True, use_mask_token=False, final_upsample="expand_first"):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{}".format(
                config.depths,
                config.depths_decoder, config.drop_path_rate))

        self.config = config
        self.final_upsample = final_upsample
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        # self.patches_resolution = self.embeddings.patch_grid

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.layernorm_up = nn.LayerNorm(config.embed_dim)

        self.embeddings = Swinv2Embeddings(config, use_mask_token=use_mask_token)
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)
        self.decoder = Swinv2Decoder(config)

        # self.num_classes = num_classes
        # self.num_layers = len(depths)
        # self.embed_dim = embed_dim
        # self.ape = ape
        # self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # self.num_features_up = int(embed_dim * 2)
        # self.mlp_ratio = mlp_ratio
        # self.final_upsample = final_upsample

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # absolute position embedding

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers): # self.num_layers = 4
        #     layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
        #                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
        #                                          patches_resolution[1] // (2 ** i_layer)),
        #                        depth=depths[i_layer],
        #                        num_heads=num_heads[i_layer],
        #                        window_size=window_size,
        #                        mlp_ratio=self.mlp_ratio,
        #                        qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                        drop=drop_rate, attn_drop=attn_drop_rate,
        #                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #                        norm_layer=norm_layer,
        #                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        #                        use_checkpoint=use_checkpoint)
        #     self.layers.append(layer)

        # build decoder layers
        # self.layers_up = nn.ModuleList()
        # self.concat_back_dim = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                               int(embed_dim * 2 ** (
        #                                           self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
        #     if i_layer == 0:
        #         layer_up = PatchExpand(
        #             input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                               patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #             dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
        #     else:
        #         layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                  input_resolution=(
        #                                  patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                  patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                                  depth=depths[(self.num_layers - 1 - i_layer)],
        #                                  num_heads=num_heads[(self.num_layers - 1 - i_layer)],
        #                                  window_size=window_size,
        #                                  mlp_ratio=self.mlp_ratio,
        #                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                  drop=drop_rate, attn_drop=attn_drop_rate,
        #                                  drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
        #                                      depths[:(self.num_layers - 1 - i_layer) + 1])],
        #                                  norm_layer=norm_layer,
        #                                  upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
        #                                  use_checkpoint=use_checkpoint)
        #     self.layers_up.append(layer_up)
        #     self.concat_back_dim.append(concat_linear)
        #
        # self.norm = norm_layer(self.num_features)
        # self.norm_up = norm_layer(self.embed_dim)


        """
        Expand 4x to get the segmentation output
        """
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(config.image_size // config.patch_size,
                                                            config.image_size // config.patch_size),
                                          dim_scale=4, dim=config.embed_dim)
            self.output = nn.Conv2d(in_channels=config.embed_dim, out_channels=config.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x, head_mask=None, bool_masked_pos=None):
        # x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        # x = self.pos_drop(x)
        # x_downsample = []
        #
        # for layer in self.layers:
        #     x_downsample.append(x)
        #     x = layer(x)
        #
        # x = self.norm(x)  # B L C
        #
        # return x, x_downsample
        # head_mask = self.get_head_mask(head_mask, len(self.config.depths))
        embedding_output, input_dimensions = self.embeddings(x, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            # head_mask=head_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        # embedding_output (B, (W/4*H/4), C)
        # encoder_outputs[1][:-1]:
        # (
            # (B, (B, (W/8*H/8), 2C))
            # (B, (W/16*H/16), 4C)
        # )
        bottleneck_output = self.layernorm(encoder_outputs[0])

        return bottleneck_output, encoder_outputs[1][:-2]

    def forward_up_features(self, x, concat_feature_list):

        _, N, _ = x.shape # (B, W/32 * H/32, 8C)
        decoder_output = self.decoder(
            hidden_states=x,
            # input_dimensions=int(math.sqrt(N)),
            concat_feature_list=concat_feature_list # type tuple (outputs from encoder blocks)
        )
        norm_output = self.layernorm_up(decoder_output) # B (W/4 * H/4) C

        return norm_output
        # parameters in decoder
        # hidden_states: torch.Tensor,
        # input_dimensions: Tuple[int, int],  # (H/16, W/16) or (patch_resolution/4, patch_resolution/4)
        # concat_feature_list,

    def forward(self, x):
        x, concat_feature_list = self.forward_features(x)
        x = self.forward_up_features(x, concat_feature_list)
        x = self.up_x4(x)

        return x

    def up_x4(self, x):
        # the size of input tensor
        H, W = self.embeddings.patch_grid
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    # forward function
    # def forward(self, x):
    #     x, x_downsample = self.forward_features(x)
    #     x, _ = self.forward_up_features(x, x_downsample)
    #     x = self.up_x4(x)
    #
    #     return x

    # Dencoder and Skip connection
    # def forward_up_features(self, x, x_downsample):
    #     for inx, layer_up in enumerate(self.layers_up):
    #         if inx == 0:
    #             x = layer_up(x)
    #         else:
    #             x = torch.cat([x, x_downsample[3 - inx]], -1)
    #             x = self.concat_back_dim[inx](x)
    #             x = layer_up(x)
    #
    #     x = self.norm_up(x)  # B L C
    #
    #     return x

    # def forward_up_features(self, x, x_downsample):
    #
    #     return_layer_list = []
    #
    #     for inx, layer_up in enumerate(self.layers_up):
    #         if inx == 0:
    #             x = layer_up(x)
    #         else:
    #             x = torch.cat([x, x_downsample[3 - inx]], -1)
    #             x = self.concat_back_dim[inx](x)
    #             x = layer_up(x)
    #
    #         return_layer_list.append(x)
    #
    #     x = self.norm_up(x)  # B L C
    #
    #     return x, return_layer_list

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops