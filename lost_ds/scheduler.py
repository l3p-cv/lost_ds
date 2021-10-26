import multiprocessing as mp 

from dask.distributed import Client, LocalCluster

def get_dask_client(client, cluster):
    if client is None:
        if cluster is None:
            cluster = LocalCluster(processes=False, asynchronous=True, 
                                   n_workers=mp.cpu_count())
        client = Client(cluster)
    elif 'Client' in str(client):
        client = client
    else:
        raise ValueError('Client {} is not supported!'.format(client))
    return client