import community
import networkx as nx
import numpy as np
import pandas as pd

from matplotlib import pylab as pl


def construct_graph(df_emails):
    '''From an email dataframe, output the biggest connected component of the
        graph where nodes are sender/recipients and edges are emails sent
        between them.
        Arguments:
            - df_emails (pd dataframe): the pd dataframe containing the emails
        Ourput:
            - G_connected (nx graph): biggest connected component of the graph
    '''
    G = nx.Graph()
    for index, row in df_emails.iterrows():
        sender = row["sender"]
        G.add_node(sender)
        recipients = row["recipients"]
        recipients_list = recipients.split()
        for rec in recipients_list:
            if "@" in rec:
                G.add_node(rec)
                G.add_edge(sender, rec, weight=1/len(recipients_list))

    if not nx.is_connected(G):
        G_connected = next(nx.connected_component_subgraphs(G))
    else:
        G_connected = G
    return G_connected


def compute_teams(G):
    '''Split G into cluster according to Louvain algorithm
        Arguments:
            - G: the email graph
        Ourput:
            dictionary with to elements:
                - teams (list): team assignment for nodes in G.
                - n_classes (int): number of clusters
    '''
    parts = community.best_partition(G)
    teams = {}
    for node in G.nodes():
        teams[node] = parts.get(node)
    n_clusters = len(set(teams.values()))
    output_dict = {}
    output_dict["n_clusters"] = n_clusters
    output_dict["teams"] = teams
    output_dict["parts"] = parts
    return output_dict


def assign_team(teams, n_clusters, email):
    '''Construct a vector to figure out in which team 'email' is.
        Arguments:
            - teams: assignmenet from Louvain algo.
            - n_clusters: number of clusters in the Louvain assignment
            - email (str): email address (one of the nodes of G)
        Ourput:
            - assignment (np.array): team assigned.
    '''
    assignment = np.zeros(n_clusters)
    assignment[teams[email]] = 1
    return assignment


def compute_summary_graph(G, n_clusters, parts):
    '''Construct a summary graph to see the weights between classes and internal
        Arguments:
            - G: graph where each node is assigned to a team in 'parts'.
            - n_clusters: number of clusters in the Louvain assignment
        Ourput:
            - G_community (nx graph): team assigned.
    '''
    G_community = nx.Graph()
    for node in G.nodes():
        value = parts.get(node)
        G_community.add_node(value)
    for source_email, sink_email in G.edges():
        source_value = parts.get(source_email)
        sink_value = parts.get(sink_email)
        if G_community.has_edge(source_value, sink_value):
            # we added this one before, just increase the weight by one
            G_community[source_value][sink_value]['weight'] += G.get_edge_data(
                source_email,
                sink_email
                )['weight']
        else:
            # new edge. add with weight=weight in G
            G_community.add_edge(
                source_value,
                sink_value,
                weight=G.get_edge_data(source_email, sink_email)['weight'])
    # renaming nodes with their intern weight
    relabeling_dict = {}
    for node in G_community.nodes():
        relabeling_dict[node] = int(G_community[node][node]['weight'])
    G_community = nx.relabel_nodes(G_community, relabeling_dict)
    return G_community
