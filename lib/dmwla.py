import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
import faiss

from sklearn.cluster import KMeans
from torch.distributions import Categorical


class DynamicMemoryWeightsLossAttention(nn.Module):

    def __init__(self, k, ch, which_conv, pool_size_per_cluster, num_k, feature_dim, warmup_total_iter=1000,
                 cp_momentum=0.3, \
                 cp_phi_momentum=0.95, device='cuda', normalization="together"):
        super(DynamicMemoryWeightsLossAttention, self).__init__()
        self.myid = "atten_concept_prototypes"
        self.device = device
        self.pool_size_per_cluster = pool_size_per_cluster
        self.num_k = num_k
        self.k = k
        self.feature_dim = feature_dim
        self.ch = ch
        self.total_pool_size = self.num_k * self.pool_size_per_cluster

        self.register_buffer('concept_pool', torch.rand(self.feature_dim, self.total_pool_size))
        self.register_buffer('concept_proto', torch.rand(self.feature_dim, self.num_k))



        self.register_buffer('warmup_iter_counter', torch.FloatTensor([0.]))
        self.warmup_total_iter = warmup_total_iter
        self.register_buffer('pool_structured', torch.FloatTensor(
            [0.]))


        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi = self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi_k = [self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0,
            bias=False).cuda()]

        for param_phi, param_phi_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
            param_phi_k.data.copy_(param_phi.data)
            param_phi_k.requires_grad = False

        self.g = self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.feature_dim, self.ch, kernel_size=1, padding=0, bias=False)



        self.cp_momentum = cp_momentum
        self.cp_phi_momentum = cp_phi_momentum


    @torch.no_grad()
    def _update_pool(self, index, content):
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim


        self.concept_pool[:, index] = content.clone()

    @torch.no_grad()
    def _update_prototypes(self, index, content):
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim

        self.concept_proto[:, index] = content.clone()

    @torch.no_grad()
    def computate_prototypes(self):
        assert not self._get_warmup_state(), f"still in warm up state {self.warmup_state}, computing prototypes is forbidden"
        self.concept_proto = self.concept_pool.detach().clone().reshape(self.feature_dim, self.num_k,
                                                                        self.pool_size_per_cluster).mean(2)

    @torch.no_grad()
    def forward_update_pool(self, activation, cluster_num, momentum=None):

        if not momentum:
            momentum = 1.
        assert cluster_num.max() < self.num_k
        q = []
        for cluster in range(self.num_k):
            cluster_concept_pool = self.concept_pool.detach().clone()[
                :, cluster * self.pool_size_per_cluster: (cluster + 1) * self.pool_size_per_cluster].cuda()
            part_concept_pool = F.softmax(cluster_concept_pool, dim=-1)
            part_concept_pool = torch.sum(part_concept_pool, dim=0)
            index_1 = torch.topk(part_concept_pool, k=int(self.k), dim=-1, largest=False)[1]

            cluster_index = torch.where(cluster_num == cluster)[0]
            if cluster_index.shape[0] == 0:
                q.append(cluster_concept_pool)
            else:
                if cluster_index.shape[0] < int(self.k):
                    repeat_times = (int(self.k) // cluster_index.shape[0]) + 1
                    replicated_index = torch.cat([cluster_index for _ in range(repeat_times)])
                    cluster_index = replicated_index[torch.randperm(replicated_index.shape[0])][: int(self.k)]
                cluster_activation = activation.T[:, cluster_index].cuda()
                cluster_updata = F.softmax(cluster_activation, dim=-1)
                cluster_updata = torch.sum(cluster_updata, dim=0)
                index_2 = torch.topk(cluster_updata, k=int(self.k), dim=-1, largest=True)[1]
                cluster_concept_pool[:, index_1] = (1. - momentum) * cluster_concept_pool[
                    :, index_1.cuda()] + momentum * cluster_activation[:, index_2.cuda()]
                q.append(cluster_concept_pool)
        self.concept_pool = torch.cat(q, dim=-1)


    @torch.no_grad()
    def pool_kmean_init_gpu(self, seed=0, gpu_num=0, temperature=1):

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k

        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 100
        clus.nredo = 10
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_num
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)
        im2cluster = [int(n[0]) for n in I]


        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)


        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])


        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d


        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))
        print(density.mean())
        density = temperature * density / density.mean()


        centroids = torch.Tensor(centroids)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster)
        density = torch.Tensor(density)

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

        del cfg, res, index, clus


        self.structure_memory_bank(results)
        print("Finish kmean init...")
        del results

    @torch.no_grad()
    def pool_kmean_init(self, seed=0, gpu_num=0, temperature=1):

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k

        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(x)

        centroids = torch.Tensor(kmeans.cluster_centers_)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(kmeans.labels_)

        results['centroids'].append(centroids)
        results['im2cluster'].append(im2cluster)


        self.structure_memory_bank(results)
        print("Finish kmean init...")

    @torch.no_grad()
    def structure_memory_bank(self, cluster_results):
        centeriod = cluster_results['centroids'][0]
        cluster_assignment = cluster_results['im2cluster'][0]

        mem_index = torch.zeros(
            self.total_pool_size).long()
        memory_states = torch.zeros(self.num_k, ).long()
        memory_cluster_insert_ptr = torch.zeros(self.num_k, ).long()


        for idx, i in enumerate(cluster_assignment):
            cluster_num = i
            if memory_states[cluster_num] == 0:


                mem_index[cluster_num * self.pool_size_per_cluster + memory_cluster_insert_ptr[cluster_num]] = idx

                memory_cluster_insert_ptr[cluster_num] += 1
                if memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster:
                    memory_states[cluster_num] = 1 - memory_states[cluster_num]
            else:

                assert memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster


        not_fill_cluster = torch.where(memory_states == 0)[0]
        print(f"memory_states {memory_states}")
        print(f"memory_cluster_insert_ptr {memory_cluster_insert_ptr}")
        for unfill_cluster in not_fill_cluster:
            cluster_ptr = memory_cluster_insert_ptr[unfill_cluster]
            assert cluster_ptr != 0, f"cluster_ptr {cluster_ptr} is zero!!!"
            existed_index = mem_index[
                unfill_cluster * self.pool_size_per_cluster: unfill_cluster * self.pool_size_per_cluster + cluster_ptr]
            print(f"existed_index {existed_index}")
            print(f"cluster_ptr {cluster_ptr}")
            print(f"(self.pool_size_per_cluster {self.pool_size_per_cluster}")
            replicate_times = (self.pool_size_per_cluster // cluster_ptr) + 1  # with more replicate and cutoff
            print(f"replicate_times {replicate_times}")
            replicated_index = torch.cat([existed_index for _ in range(replicate_times)])
            print(f"replicated_index {replicated_index}")
            replicated_index = replicated_index[torch.randperm(replicated_index.shape[0])][
                :self.pool_size_per_cluster]

            assert replicated_index.shape[
                       0] == self.pool_size_per_cluster, f"replicated_index ({replicated_index.shape}) should has the same len as pool_size_per_cluster ({self.pool_size_per_cluster})"
            mem_index[unfill_cluster * self.pool_size_per_cluster: (
                                                                               unfill_cluster + 1) * self.pool_size_per_cluster] = replicated_index

            memory_cluster_insert_ptr[unfill_cluster] = self.pool_size_per_cluster

            memory_states[unfill_cluster] = 1

        assert (memory_states == 0).sum() == 0, f"memory_states has zeros: {memory_states}"
        assert (
                           memory_cluster_insert_ptr != self.pool_size_per_cluster).sum() == 0, f"memory_cluster_insert_ptr didn't match with pool_size_per_cluster: {memory_cluster_insert_ptr}"


        self._update_pool(torch.arange(mem_index.shape[0]), self.concept_pool[:, mem_index])

        self._update_prototypes(torch.arange(self.num_k), centeriod.T.cuda())
        print(f"Concept pool updated by kmeans clusters...")

    def _check_warmup_state(self):

        if self.warmup_iter_counter == self.warmup_total_iter:

            self.pool_kmean_init()

    def warmup_sampling(self, x):
        n, c, h, w = x.shape
        assert self._get_warmup_state(), "calling warmup sampling when warmup state is 0"


        sample_per_instance = max(int(self.total_pool_size / n), 1)

        index = torch.randint(h * w, size=(n, 1, sample_per_instance)).repeat(1, c, 1).to(
            self.device)
        sampled_columns = torch.gather(x.reshape(n, c, h * w), 2, index)
        sampled_columns = torch.transpose(sampled_columns, 1, 0).reshape(c,
                                                                         -1).contiguous()

        percentage = (self.warmup_iter_counter + 1) / self.warmup_total_iter * 0.5
        print(f"percentage {percentage.item()}")
        sample_column_num = max(1, int(percentage * sampled_columns.shape[1]))
        sampled_columns_idx = torch.randint(sampled_columns.shape[1], size=(sample_column_num,))
        sampled_columns = sampled_columns[:, sampled_columns_idx]


        update_idx = torch.randperm(self.concept_pool.shape[1])[:sample_column_num]
        self._update_pool(update_idx, sampled_columns)


        self.warmup_iter_counter += 1


    def forward(self, x, device="cuda", evaluation=False):

        if self._get_warmup_state():
            print(
                f"Warmup state? {self._get_warmup_state()} self.warmup_iter_counter {self.warmup_iter_counter.item()} self.warmup_total_iter {self.warmup_total_iter}")
            theta = self.theta(x)
            phi = self.phi(x)
            g = self.g(x)

            n, c, h, w = theta.shape


            self.warmup_sampling(phi)
            self._check_warmup_state()


            theta = theta.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
            phi = phi.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
            g = g.view(-1, self.feature_dim, x.shape[2] * x.shape[3])


            beta = F.softmax(torch.bmm(theta.transpose(1, 2).contiguous(), phi), -1)


            o = self.o(torch.bmm(g, beta.transpose(1, 2).contiguous()).view(-1,
                                                                            self.feature_dim, x.shape[2], x.shape[3]))


            return o + x

        else:


            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)
            g = self.g(x)  # [n, c, h, w]
            n, c, h, w = theta.shape

            theta = torch.transpose(torch.transpose(theta, 0, 1).reshape(c, n * h * w), 0,
                                    1).contiguous()
            phi = torch.transpose(torch.transpose(phi, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()
            g = torch.transpose(torch.transpose(g, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()
            with torch.no_grad():
                theta_atten_proto = nn.CosineSimilarity(dim=-1)(theta.unsqueeze(-2),
                                                                self.concept_proto.detach().clone().T.unsqueeze(
                                                                    -3))
                cluster_affinity = F.softmax(theta_atten_proto, dim=1)
                cluster_assignment = cluster_affinity.max(1)[1]



            dot_product = []
            cluster_indexs = []

            for cluster in range(self.num_k):
                cluster_index = torch.where(cluster_assignment == cluster)[0]
                theta_cluster = theta[cluster_index]

                cluster_pool = self.concept_pool.detach().clone()[:, cluster * self.pool_size_per_cluster: (
                                                                                                                       cluster + 1) * self.pool_size_per_cluster]

                theta_cluster_attend_weight = torch.matmul(theta_cluster,
                                                           cluster_pool)

                dot_product.append(theta_cluster_attend_weight)
                cluster_indexs.append(cluster_index)


            dot_product = torch.cat(dot_product, axis=0)
            cluster_indexs = torch.cat(cluster_indexs, axis=0)

            mapping_to_normal_index = torch.argsort(cluster_indexs)
            similarity_clusters = dot_product[mapping_to_normal_index]

            similarity_context = torch.bmm(theta.reshape(n, h * w, c),
                                           torch.transpose(phi.reshape(n, h * w, c), 1, 2))
            similarity_context = similarity_context.reshape(n * h * w, h * w)
            atten_weight = torch.cat([similarity_clusters, similarity_context],
                                     axis=1)


            pool_residuals = []
            cluster_indexs = []
            for cluster in range(self.num_k):
                cluster_index = torch.where(cluster_assignment == cluster)[0]
                theta_cluster = theta[cluster_index]
                atten_weight_pool_cluster = atten_weight[
                    cluster_index, :self.pool_size_per_cluster]
                atten_weight_pool_cluster = F.softmax(atten_weight_pool_cluster, dim=1)

                cluster_pool = self.concept_pool.detach().clone()[:, cluster * self.pool_size_per_cluster: (
                                                                                                                       cluster + 1) * self.pool_size_per_cluster]
                pool_residual = torch.matmul(atten_weight_pool_cluster,
                                             cluster_pool.T)
                pool_residuals.append(pool_residual)
                cluster_indexs.append(cluster_index)
            pool_residuals = torch.cat(pool_residuals, axis=0)
            cluster_indexs = torch.cat(cluster_indexs, axis=0)

            mapping_to_normal_index = torch.argsort(cluster_indexs)
            pool_residuals = pool_residuals[mapping_to_normal_index]
            pool_residuals = pool_residuals.reshape(n, h * w, c)

            atten_weight_context = atten_weight[:, self.pool_size_per_cluster:]
            atten_weight_context = F.softmax(atten_weight_context, dim=1)
            atten_weight_context = atten_weight_context.reshape(n, h * w, h * w)
            context_residuals = torch.bmm(atten_weight_context, g.reshape(n, h * w,
                                                                          c))

            beta_residual = pool_residuals + context_residuals
            beta_residual = torch.transpose(beta_residual, 1, 2).reshape(n, c, h, w).contiguous()

            o = self.o(beta_residual)

            with torch.no_grad():
                phi_k = self.phi_k[0](x)
                phi_k = torch.transpose(torch.transpose(phi_k, 0, 1).reshape(c, n * h * w), 0,
                                        1).contiguous()
                phi_k_atten_proto = torch.matmul(phi_k, self.concept_proto.detach().clone())
                phi_k_atten_proto = phi_k_atten_proto.reshape(n, h * w, -1)
                cluster_affinity_phi_k = F.softmax(phi_k_atten_proto, dim=2)
                cluster_assignment_phi_k = cluster_affinity_phi_k.max(2)[1].reshape(n * h * w, )


                self.forward_update_pool(phi_k, cluster_assignment_phi_k, momentum=self.cp_momentum)


                self.computate_prototypes()


                for param_q, param_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
                    param_k.data = param_k.data * self.cp_phi_momentum + param_q.data * (1. - self.cp_phi_momentum)

            if evaluation:
                return o + x, cluster_affinity
            return o + x



    def get_cluster_num_index(self, idx):
        assert idx < self.total_pool_size
        return idx // self.pool_size_per_cluster

    def get_cluster_ptr(self, cluster_num):
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        return self.pool_size_per_cluster * cluster_num

    def _get_warmup_state(self):

        return self.warmup_iter_counter.cpu() <= self.warmup_total_iter


