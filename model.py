from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

class SimpleGraph:
    """
    Estrutura simplificada para armazenar nós, arestas e embeddings.
    """
    def __init__(self):
        self.node_states = {}   # dict: node_id -> Tensor
        self.node_activations = {}  # dict: node_id -> Tensor
        self.edges = set()      # set of (src, dst)

    def num_nodes(self):
        return len(self.node_states)

    def add_node(self, hv, activation):
        idx = len(self.node_states)
        self.node_states[idx] = hv
        self.node_activations[idx] = activation
        return idx

    def add_edge(self, src, dst):
        self.edges.add((src, dst))
        self.edges.add((dst, src))  # assumindo grafo não-direcional

    def has_edge_between(self, src, dst):
        return (src, dst) in self.edges

class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()
        self.graph_hidden_size = 2 * node_hidden_size

        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)

    def forward(self, g):
        if g.num_nodes() == 0:
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Coleta embeddings dos nós
            hvs = torch.stack([h for h in g.node_states.values()], dim=0)
            return (self.node_gating(hvs) * self.node_to_graph(hvs)).sum(
                0, keepdim=True
            )
        
class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds
        self.node_activation_hidden_size = 2 * node_hidden_size

        # MLPs para gerar mensagens
        self.message_funcs = nn.ModuleList([
            nn.Linear(2 * node_hidden_size, self.node_activation_hidden_size)
            for _ in range(num_prop_rounds)
        ])

        # GRUCells para atualizar embeddings dos nós
        self.node_update_funcs = nn.ModuleList([
            nn.GRUCell(self.node_activation_hidden_size, node_hidden_size)
            for _ in range(num_prop_rounds)
        ])

    def forward(self, g):
        if len(g.edges) == 0:
            return

        for t in range(self.num_prop_rounds):
            # 1. Mensagens por aresta
            messages = {v: [] for v in g.node_states}

            for (src, dst) in g.edges:
                h_src = g.node_states[src]
                h_dst = g.node_states[dst]
                # Concatenar embeddings
                m_input = torch.cat([h_src, h_dst], dim=0)
                m = self.message_funcs[t](m_input)
                messages[dst].append(m)

            # 2. Agregação por nó
            for v in messages:
                if messages[v]:
                    agg = torch.sum(torch.stack(messages[v]), dim=0)
                    h_prev = g.node_states[v]
                    h_new = self.node_update_funcs[t](agg.unsqueeze(0), h_prev.unsqueeze(0))
                    g.node_states[v] = h_new.squeeze(0)

class AddNode(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddNode, self).__init__()

        self.graph_op = {"embed": graph_embed_func}

        self.stop = 1  # classe 1 significa "parar"
        self.add_node = nn.Linear(graph_embed_func.graph_hidden_size, 1)

        self.node_type_embed = nn.Embedding(1, node_hidden_size)
        self.initialize_hv = nn.Linear(
            node_hidden_size + graph_embed_func.graph_hidden_size,
            node_hidden_size,
        )

        self.init_node_activation = torch.zeros(2 * node_hidden_size)

    def _initialize_node_repr(self, g, node_type, graph_embed):
        hv_init = self.initialize_hv(
            torch.cat([
                self.node_type_embed(torch.LongTensor([node_type])),
                graph_embed
            ], dim=1)
        ).squeeze(0)
        g.add_node(hv_init, self.init_node_activation)

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op["embed"](g)
        logit = self.add_node(graph_embed)

        if action is None and not self.training:
            action = Bernoulli(logits=logit).sample().item()  # usa logits, mais estável

        stop = bool(action == self.stop)
        if not stop:
            self._initialize_node_repr(g, action, graph_embed)

        if self.training:
            sample_log_prob = F.logsigmoid(-logit) if action == 0 else F.logsigmoid(logit)
            self.log_prob.append(sample_log_prob)
        return stop

class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddEdge, self).__init__()

        self.graph_op = {"embed": graph_embed_func}
        self.add_edge = nn.Linear(
            graph_embed_func.graph_hidden_size + node_hidden_size, 1
        )

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op["embed"](g)
        src_embed   = g.node_states[g.num_nodes() - 1].unsqueeze(0)
        logit = self.add_edge(torch.cat([graph_embed, src_embed], dim=1))

        if action is None and not self.training:
            action = Bernoulli(logits=logit).sample().item()

        to_add = bool(action == 0)
        if self.training:
            self.log_prob.append(F.logsigmoid(-logit) if action == 0 else F.logsigmoid(logit))
        return to_add

class ChooseDestAndUpdate(nn.Module):
    def __init__(self, graph_prop_func, node_hidden_size):
        super(ChooseDestAndUpdate, self).__init__()

        self.graph_op = {"prop": graph_prop_func}
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, dest=None):
        src = g.num_nodes() - 1
        possible = list(range(src))
        if not possible:
            return
        device = next(self.parameters()).device
        src_embed   = g.node_states[src].unsqueeze(0).repeat(len(possible), 1).to(device)
        dests_embed = torch.stack([g.node_states[d] for d in possible], dim=0).to(device)

        cat = torch.cat([dests_embed, src_embed], dim=1)
        scores = self.choose_dest(cat).view(1, -1)
        probs  = F.softmax(scores, dim=1)

        if dest is None and not self.training:
            dest = Categorical(probs).sample().item()

        dest_node = possible[dest]
        if not g.has_edge_between(src, dest_node):
            g.add_edge(src, dest_node)
            self.graph_op["prop"](g)

        if self.training:
            self.log_prob.append(F.log_softmax(scores, dim=1)[:, dest:dest+1])

class DGMG(nn.Module):
    def __init__(self, v_max, node_hidden_size, num_prop_rounds, generation_mode: str = "general"):
        super(DGMG, self).__init__()

        self.v_max = v_max
        self.generation_mode = generation_mode

        self.graph_embed = GraphEmbed(node_hidden_size)
        self.graph_prop = GraphProp(num_prop_rounds, node_hidden_size)

        self.add_node_agent = AddNode(self.graph_embed, node_hidden_size)
        self.add_edge_agent = AddEdge(self.graph_embed, node_hidden_size)
        self.choose_dest_agent = ChooseDestAndUpdate(
            self.graph_prop, node_hidden_size
        )

        self.init_weights()

    def init_weights(self):
        from utils import dgmg_message_weight_init, weights_init

        self.graph_embed.apply(weights_init)
        self.graph_prop.apply(weights_init)
        self.add_node_agent.apply(weights_init)
        self.add_edge_agent.apply(weights_init)
        self.choose_dest_agent.apply(weights_init)

        self.graph_prop.message_funcs.apply(dgmg_message_weight_init)

    @property
    def action_step(self):
        old_step_count = self.step_count
        self.step_count += 1
        return old_step_count

    def prepare_for_train(self):
        self.step_count = 0
        self.add_node_agent.prepare_training()
        self.add_edge_agent.prepare_training()
        self.choose_dest_agent.prepare_training()

    def add_node_and_update(self, a=None):
        return self.add_node_agent(self.g, a)

    def add_edge_or_not(self, a=None):
        return self.add_edge_agent(self.g, a)

    def choose_dest_and_update(self, a=None):
        self.choose_dest_agent(self.g, a)

    def get_log_prob(self):
        return (
            torch.cat(self.add_node_agent.log_prob).sum()
            + torch.cat(self.add_edge_agent.log_prob).sum()
            + torch.cat(self.choose_dest_agent.log_prob).sum()
        )

    def forward_train(self, actions):
        self.prepare_for_train()
        stop = self.add_node_and_update(a=actions[self.action_step])

        while not stop:
            to_add_edge = self.add_edge_or_not(a=actions[self.action_step])

            while to_add_edge:
                self.choose_dest_and_update(a=actions[self.action_step])
                to_add_edge = self.add_edge_or_not(a=actions[self.action_step])

            stop = self.add_node_and_update(a=actions[self.action_step])

        return self.get_log_prob()

    def forward_inference(self):
    # força adicionar o primeiro nó
        _ = self.add_node_and_update(a=0)  # 0 = add

        while (self.g.num_nodes() < self.v_max):
            if self.generation_mode == "tree":
                to_add_edge = self.add_edge_or_not()
                if to_add_edge and self.g.num_nodes() > 1:
                    self.choose_dest_and_update()
                # não repetimos: máx. 1 aresta por nó
            else:
                num_trials = 0
                to_add_edge = self.add_edge_or_not()
                while to_add_edge and (num_trials < self.g.num_nodes() - 1):
                    self.choose_dest_and_update()
                    num_trials += 1
                    to_add_edge = self.add_edge_or_not()

            stop = self.add_node_and_update()  # aqui pode amostrar parar
            if stop:
                break
        return self.g

    def forward(self, actions=None):
        self.g = SimpleGraph()
        if self.training:
            return self.forward_train(actions)
        else:
            return self.forward_inference()
