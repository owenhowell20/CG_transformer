import torch
from src.performer_pytorch import softmax_kernel
import torch.nn.functional as F


def exact_attention(Q, K, V):
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d**0.5)  # [B, H, L, L]
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def test_softmax_kernel_permutations():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    num_heads = 8
    seq_len = 256
    num_random_features = 256
    dim = 8
    eps = 1e-3

    # Create a random permutation of the seq_len dimension
    permutation = torch.randperm(seq_len, device=device)

    queries = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
    keys = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
    values = torch.randn(batch_size, num_heads, seq_len, dim, device=device)

    ### Apply the permutation to the seq_len dimension
    queries_permuted = queries[:, :, permutation]
    keys_permuted = keys[:, :, permutation]
    values_permuted = values[:, :, permutation]

    projection_matrix = torch.randn(num_random_features, dim)  # e.g., (256, 64)

    phi_queries = softmax_kernel(
        queries, projection_matrix=projection_matrix, is_query=True
    )
    phi_keys = softmax_kernel(keys, projection_matrix=projection_matrix, is_query=False)

    phi_queries_permuted = softmax_kernel(
        queries_permuted, projection_matrix=projection_matrix, is_query=True
    )
    phi_keys_permuted = softmax_kernel(
        keys_permuted, projection_matrix=projection_matrix, is_query=False
    )

    # key values contraction
    kv = torch.einsum("bhld,bhlm->bhdm", phi_keys, values)  # [B, H, F, D]

    # denomonator
    z = 1 / (
        torch.einsum("bhld,bhd->bhl", phi_queries, phi_keys.sum(dim=2)) + eps
    )  # [B, H, L]

    # Compute numerator
    output = torch.einsum("bhld,bhdm->bhlm", phi_queries, kv)  # [B, H, L, D]

    # Step 4: Normalize
    output = output * z.unsqueeze(-1)  # [B, H, L, D]

    # key values contraction
    kv_permuted = torch.einsum(
        "bhld,bhlm->bhdm", phi_keys_permuted, values_permuted
    )  # [B, H, F, D]

    # denomonator
    z_permuted = 1 / (
        torch.einsum(
            "bhld,bhd->bhl", phi_queries_permuted, phi_keys_permuted.sum(dim=2)
        )
        + eps
    )  # [B, H, L]

    # Compute numerator
    output_permuted = torch.einsum(
        "bhld,bhdm->bhlm", phi_queries_permuted, kv_permuted
    )  # [B, H, L, D]

    # Step 4: Normalize
    output_permuted = output_permuted * z_permuted.unsqueeze(-1)  # [B, H, L, D]

    ### check if output is permuted correctly?
    permuted_output = output[:, :, permutation]

    # Shape assertion
    assert output_permuted.shape == permuted_output.shape

    # Check for closeness
    assert torch.allclose(
        output_permuted, permuted_output, atol=1e-2, rtol=1e-2
    ), "Output does not match exact attention."


def test_exact_permutations():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    num_heads = 8
    seq_len = 256
    num_random_features = 256
    dim = 8
    eps = 1e-3

    # Create a random permutation of the seq_len dimension
    permutation = torch.randperm(seq_len, device=device)

    queries = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
    keys = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
    values = torch.randn(batch_size, num_heads, seq_len, dim, device=device)

    ### Apply the permutation to the seq_len dimension
    queries_permuted = queries[:, :, permutation]
    keys_permuted = keys[:, :, permutation]
    values_permuted = values[:, :, permutation]

    # Step 4: Normalize
    output = exact_attention(queries, keys, values)  # [B, H, L, D]

    # Step 4: Normalize
    output_permuted = exact_attention(queries_permuted, keys_permuted, values_permuted)

    ### check if output is permuted correctly?
    permuted_output = output[:, :, permutation]

    # Shape assertion
    assert output_permuted.shape == permuted_output.shape

    # Check for closeness
    assert torch.allclose(
        output_permuted, permuted_output, atol=1e-2, rtol=1e-2
    ), "Output does not match exact attention."


# #
#
#
# def test_softmax_kernel():
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     batch_size = 1
#     num_heads = 1
#     seq_len = 256
#     num_random_features = 200
#     dim = 1
#     eps = 1e-3
#
#     queries = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
#     keys = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
#     values = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
#
#     projection_matrix = torch.randn(num_random_features, dim)  # e.g., (256, 64)
#
#     phi_queries = softmax_kernel(
#         queries, projection_matrix=projection_matrix, is_query=True
#     )
#     phi_keys = softmax_kernel(keys, projection_matrix=projection_matrix, is_query=False)
#
#     exact = exact_attention(queries, keys, values)
#
#     # key values contraction
#     kv = torch.einsum("bhld,bhlm->bhdm", phi_keys, values)  # [B, H, F, D]
#
#     # denomonator
#     z = 1 / (
#         torch.einsum("bhld,bhd->bhl", phi_queries, phi_keys.sum(dim=2)) + eps
#     )  # [B, H, L]
#
#     # Compute numerator
#     output = torch.einsum("bhld,bhdm->bhlm", phi_queries, kv)  # [B, H, L, D]
#
#     # Step 4: Normalize
#     output = output * z.unsqueeze(-1)  # [B, H, L, D]
#
#     # Shape assertion
#     assert (
#         exact.shape == output.shape
#     ), f"Shape mismatch: {exact.shape} vs {output.shape}"
#
#     # Check for closeness
#     assert torch.allclose(
#         exact, output, rtol=1
#     ), "Output does not match exact attention."
#
#     # diff = exact - output
#     # plt.imshow(diff.cpu().numpy()[0, 0], cmap='hot', interpolation='nearest')
#     # plt.colorbar()
#     # plt.show()
