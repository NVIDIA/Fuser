import os
import torch
import torch.distributed as dist


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    assert world_size >= 4, "This test requires at least 4 GPUs."
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Create a random tensor on each process
    device = torch.device(f"cuda:{rank}")
    in_tensor = torch.full((3, 4), rank, dtype=torch.float32, device=device)
    out_tensor = torch.full((3, 4), -1, dtype=torch.float32, device=device)

    for sender, receiver in ((0, 1), (2, 0), (3, 2)):
        if sender == rank:
            dist.send(in_tensor, dst=receiver)
        if receiver == rank:
            dist.recv(out_tensor, src=sender)

    # Print information on each process
    print(f"Rank: {rank}, Received tensor: {out_tensor}")

    # Cleanup (optional, call at the end)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
