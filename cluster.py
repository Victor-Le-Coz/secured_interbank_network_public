from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from IPython.display import display
import datetime
import numpy as np
import os


def new_launch_cluster(
    task_memory,
    job_walltime="1:00:00",
    max_cpu=False,
    mltp=True,
    single_node=False,
    show=False,
):
    """
    source: https://documentations.ipsl.fr/spirit/common/dask_jobqueue.html
    :param: TASK_MEMORY: integer between 1 and 1480: memory required for a single task (can be a thread or a process/job, trying to maximize the number of threads)
    :param: JOB_WALLTIME: time required for a single process
    """
    # path mgt
    # WORKER_LOG_DIRECTORY = f"./dask_logs/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" # DOESN'T WORK GENERATE WRONG MEM ALLOC !
    WORKER_LOG_DIRECTORY = f"./dask_logs/{datetime.datetime.now().strftime('D%Y-%m_%d_T%H-%M-%S.%f')}/"
    os.system(f"rm -rf ./*.out")  # reset outfiles

    # optimization for cfm constraints
    MAX_MEM_CFM = 9500  # G (dask shows dashboard in GiB, 1 GB = 0.9.3 GiB)
    MAX_CPU_CFM = 500
    CPU_PER_NODE = 76
    MEM_PER_NODE = 1480  # G

    # override for max cpu:
    if not (max_cpu):
        max_cpu = MAX_CPU_CFM

    # nb of jobs, processes, and threads
    total_cpus = min(int(MAX_MEM_CFM / task_memory), max_cpu)  #  1 to 500

    # mltp means multiprocessing only: i.e. one thread per worker, otherwise we maximize the number of thread on a single machine / node (i.e. on a single worker)
    if mltp:
        NB_THREADS_PER_WORKER = 1
    else:
        NB_THREADS_PER_WORKER = min(
            int(MEM_PER_NODE / task_memory), CPU_PER_NODE, total_cpus
        )  # from 1 to 76

    # single node means we want a single worker on a single machine (node) in multithreading
    if single_node:
        NB_WORKERS = 1
        NB_THREADS_PER_WORKER = min(
            int(MEM_PER_NODE / task_memory), CPU_PER_NODE, total_cpus
        )

    NB_WORKERS = int(total_cpus / NB_THREADS_PER_WORKER)  # from 6 to 12
    WORKER_MEM = NB_THREADS_PER_WORKER * task_memory  # from 76 tp 1480
    NB_WORKERS_PER_JOB = 1
    nb_jobs = int(np.ceil(NB_WORKERS / NB_WORKERS_PER_JOB))
    nb_cores_per_job = NB_THREADS_PER_WORKER * NB_WORKERS_PER_JOB

    # memory per job, 1 job = 1 process here, so it is also the memory per process
    job_mem = NB_WORKERS_PER_JOB * WORKER_MEM

    # create the cluster for a single node
    cluster = SLURMCluster(
        queue="compute",
        account="trainees",
        cores=nb_cores_per_job,  # this correspond to sbatch cpus-per-task
        processes=NB_WORKERS_PER_JOB,
        memory=f"{job_mem}G",  # memory of a given job
        walltime=job_walltime,
        log_directory=WORKER_LOG_DIRECTORY,
        interface="eth0",
    )

    # # cdefine the nb of jobs (better than scalling on the workers)
    cluster.scale(jobs=nb_jobs)

    # initialise the client with this cluester (this will be used by dask)
    client = Client(cluster)

    # displays the client to monitor the performance of the run
    display(client)

    if show:
        print(
            f"""
        > Cluster configuration
        - total cpus: {NB_THREADS_PER_WORKER*NB_WORKERS} (<= 500)
        - total memory: {job_mem*nb_jobs} GB (< 9500 GB)
        - memory per task : {int(WORKER_MEM/NB_THREADS_PER_WORKER)} GB (= requested {task_memory} GB)

        # Worker configuration
        - nb workers: {NB_WORKERS}
        - memory per worker: {WORKER_MEM} GB (< 1480 GB)
        - nb threads per worker: {NB_THREADS_PER_WORKER} (<= 76)

        # Cluster job configuration
        - nb jobs: {nb_jobs}
        - nb cores per job {nb_cores_per_job}
        - nb workers per job: {NB_WORKERS_PER_JOB}
        - memory per job: {job_mem} Gio
        - job walltime: {job_walltime}
        - log directory path: {WORKER_LOG_DIRECTORY}
        """
        )

    return client, cluster
