import torch
import torch.nn as nn
import torch.nn.functional as F
from neat.nn import FeedForwardNetwork
import math

class Genome:
    def __init__(self, genome, config):
        """
        Armazena o genome do NEAT (objeto neat-python)
        e a configuração do NEAT, para uso posterior.
        """
        self.genome = genome
        self.config = config
    
    def get_topological_info(self):
        """
        Cria um FeedForwardNetwork usando neat-python, 
        só para extrair as informações de topologia:
          - A lista completa em ordem topológica (node_order)
          - Quais são os IDs dos nós de entrada (input_nodes)
          - Quais são os IDs dos nós de saída (output_nodes)

        Retornamos (node_order, input_nodes, output_nodes).
        """
        net = FeedForwardNetwork.create(self.genome, self.config)
        
        # 1) 'node_order' começa com os nós de entrada do NEAT
        node_order = list(net.input_nodes)

        # 2) Depois adiciona todos os nós processados em net.node_evals (ordem topológica)
        for node_eval in net.node_evals:
            node_id = node_eval[0]
            node_order.append(node_id)
        
        # 3) Garante que todos os nós de saída (separados) estão em node_order
        for output in net.output_nodes:
            if output not in node_order:
                node_order.append(output)
        
        return node_order, net.input_nodes, net.output_nodes

    def decode_genome_to_torch(self):
        """
        Constrói dinamicamente um nn.Module PyTorch
        que respeita a topologia definida pelo NEAT.
        """

        # Extrai as listas: nós em ordem topológica, nós de entrada, nós de saída
        node_order, in_nodes, out_nodes = self.get_topological_info()

        # Certifica que todos os nós em conexões (genome.connections) 
        # estão no node_order
        for cg in self.genome.connections.values():
            if cg.enabled:
                in_node, out_node = cg.key
                if in_node not in node_order:
                    node_order.append(in_node)
                if out_node not in node_order:
                    node_order.append(out_node)

        # Montamos a classe interna NeatModule (nn.Module)
        genome = self.genome
        config = self.config
        topo_order = node_order

        class NeatModule(nn.Module):
            def __init__(_self, genome, config, node_order, in_nodes, out_nodes):
                super().__init__()
                # Guarda referências
                _self.genome = genome
                _self.config = config
                _self.node_order = node_order
                _self.input_nodes = in_nodes
                _self.output_nodes = out_nodes

                # Coleta as conexões habilitadas
                _self.connections = []
                for cg in _self.genome.connections.values():
                    if cg.enabled:
                        i_node, o_node = cg.key
                        w = cg.weight
                        _self.connections.append((i_node, o_node, w))

                # Numero de inputs/outputs do config NEAT
                _self.num_inputs = _self.config.genome_config.num_inputs
                _self.num_outputs = _self.config.genome_config.num_outputs
                
                # Cria mapa: node_id -> índice no vetor de ativações
                _self.node_id_to_idx = {}
                for i, n_id in enumerate(_self.node_order):
                    _self.node_id_to_idx[n_id] = i

            def forward(_self, x):
                """
                Executa o forward pass da topologia NEAT:
                  1. Alimenta nós de entrada
                  2. Propaga nós ocultos e de saída
                  3. Retorna tensor [batch_size, num_outputs]
                """
                batch_size = x.shape[0]
                total_nodes = len(_self.node_order)

                # Cria ativação zero para todos os nós
                activations = x.new_zeros((batch_size, total_nodes))

                # (1) Alimentar nós de entrada
                for col_index, node_id in enumerate(_self.input_nodes):
                    idx = _self.node_id_to_idx[node_id]
                    activations[:, idx] = x[:, col_index]

                # (2) Monta adjacency: para cada nó destino, 
                #     lista de (nó de origem, peso)
                adjacency = {n_id: [] for n_id in _self.node_order}
                for (i_node, o_node, w) in _self.connections:
                    # Ignora se algum nó não está em node_id_to_idx
                    if i_node not in _self.node_id_to_idx or o_node not in _self.node_id_to_idx:
                        continue
                    adjacency[o_node].append((i_node, w))

                # (3) Calcula saída dos nós (exceto entrada) na ordem topológica
                for node_id in _self.node_order[_self.num_inputs:]:
                    node_idx = _self.node_id_to_idx[node_id]
                    total_in = torch.zeros(batch_size, device=x.device)
                    for (src_id, w) in adjacency[node_id]:
                        src_idx = _self.node_id_to_idx[src_id]
                        total_in += activations[:, src_idx] * w
                    
                    # Exemplo: softmax por nó (não é comum, mas você pediu):
                    node_out = F.softmax(total_in, dim=0)
                    activations[:, node_idx] = node_out

                # (4) Extrai as saídas (na ordem do net.output_nodes do NEAT)
                outs = []
                for node_id in _self.output_nodes:
                    out_idx = _self.node_id_to_idx[node_id]
                    outs.append(activations[:, out_idx])

                # Retorna shape [batch_size, num_outputs]
                return torch.stack(outs, dim=1)

        return NeatModule(genome, config, topo_order, in_nodes, out_nodes)


def eval_genomes(genomes, config, X_train, y_train):
    """
    Função de avaliação que o NEAT chama em cada geração,
    definindo genome.fitness para cada genome.
    Usamos uma métrica que retorna 1 se a rede é perfeita,
    e 0 se a rede for equivalente a chute aleatório.
    """

    # Converte dados para tensores no device
    device = "cuda"
    X_train_torch = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long, device=device)

    # Número de classes
    num_classes = config.genome_config.num_outputs

    for genome_id, genome in genomes:
        # Monta a rede PyTorch a partir do genome
        neat_net = Genome(genome, config)
        net = neat_net.decode_genome_to_torch().to(device)

        # Forward pass no dataset (cuidado com memória para datasets grandes)
        outputs = net(X_train_torch)  # [batch_size, num_classes], já "softmaxado" por nó

        # p_correct[i] = probabilidade que a rede atribui para a classe correta
        # Adicionamos um epsilon para evitar log(0)
        eps = 1e-12
        p_correct = outputs[range(len(y_train_torch)), y_train_torch] + eps

        # mean_ll = média do log dessas probabilidades
        log_p_correct = torch.log(p_correct)
        mean_ll = log_p_correct.mean().item()

        # score = normaliza entre 0 e 1 (0 ~ chute aleatório, 1 ~ perfeito)
        # Se rede for perfeita => log(1)=0 => mean_ll=0 => score=1
        # Se rede for aleatória => log(1/k)=-log(k) => mean_ll=-log(k) => score=0
        raw_score = 1.0 + (mean_ll / math.log(num_classes))
        final_score = max(0.0, min(1.0, raw_score))

        # Atribui o fitness = final_score
        genome.fitness = final_score