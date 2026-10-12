import torch
import torch.distributed as dist


def _ring_exchange(send, recv, rank, world_size):
    requests = [
        dist.isend(send, (rank + 1) % world_size),
        dist.irecv(recv, (rank - 1) % world_size),
    ]
    for request in requests:
        request.wait()


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[recv_index])
        _ring_exchange(chunks[send_index], received, rank, world_size)
        chunks[recv_index] += received
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        _ring_exchange(chunks[send_index], chunks[recv_index], rank, world_size)
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
