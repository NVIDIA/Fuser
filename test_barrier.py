import logging
import os
import time
import torch.distributed as dist


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    assert world_size >= 3, "This test requires at least 3 GPUs."
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    for i in range(4):
        time.sleep(rank)
        logging.info(f"Rank {rank}: enter barrier {i}")
        dist.barrier()
        logging.info(f"Rank {rank}: exit barrier {i}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
