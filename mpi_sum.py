from mpi4py import MPI
import sys

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk = n // size
    remainder = n % size

    start = rank * chunk + min(rank, remainder) + 1
    end = start + chunk + (1 if rank < remainder else 0) - 1

    local_sum = sum(range(start, end + 1))

    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Celkovy soucet cisel 1 az {n} je: {total_sum}")

if __name__ == "__main__":
    main()
