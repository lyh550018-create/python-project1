"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H]."""
        # reshape splits the feature axis into (H, D/H), giving [B,N,H,D/H].
        # The H axis is still behind the token axis at this point, so transpose
        # (0,2,1,3) moves it forward; only then does each head own its slice.
        B, N, _ = x.shape
        return x.reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Q, K and V live in one [3D,D] matrix. One matmul gives [B,N,3D];
        # the last axis is then split into three equal parts.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: scores = Q K^T / sqrt(head_dim), shape [B,H,N,N].
        # Cast the scale to float32 explicitly: np.sqrt(int) returns a float64
        # scalar, which would upgrade the whole array and fail the dtype check.
        k_t = k.swapaxes(-1, -2)
        scores = (q @ k_t) / np.float32(np.sqrt(self.head_dim))

        # Kept for the text tower; the image encoder always passes causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows with the coefficients, then merge the heads
        # back: [B,H,N,D/H] -> [B,N,H,D/H] -> [B,N,D].
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.width)

        # TODO-03c: output projection. Stored weights are [out,in], so use .T.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Pre-norm attention branch, then the first residual addition.
        update = self.attention(self.norm1(x), causal=causal)
        residual = x + update

        # TODO-04: pre-norm MLP branch, 768 -> 3072 -> 768, exact GELU.
        u = self.norm2(residual)
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias

        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072]."""
        # 224 / 32 = 7, so the image is a 7x7 grid of 32x32 patches.
        # Split the spatial axes into (grid, patch size), move the two grid axes
        # in front of (C,32,32) with transpose, then flatten the last four axes.
        # Patch order is row-major; the order inside a patch is C, H, W.
        B = images.shape[0]
        patches = images.reshape(B, 3, 7, 32, 7, 32)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(B, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05: patch projection, no bias on this layer.
        patch_tokens = patches @ self.patch_weight.T

        # One CLS row per image, tiled over the batch axis.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # Prepend CLS on the token axis, then add learned positions.
        # self.position is [50,768] and broadcasts over the batch axis.
        return np.concatenate((cls, patch_tokens), axis=1) + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)  # [B,50,768]

        # Take the CLS row, apply post norm, then project to 512 dims.
        # visual.proj is stored [in,out], so it is NOT transposed (unlike the
        # linear layers above, whose weights are [out,in]).
        cls = tokens[:, 0, :]
        output = self.post_norm(cls) @ self.projection

        return unit_vector(output)
