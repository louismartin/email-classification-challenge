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
                - values (list): team assignment for nodes in G.
                - n_classes (int): number of clusters
    '''
    parts = community.best_partition(G)
    teams = {}
    for node in G.nodes():
        teams[node] = parts.get(node)
    n_clusters = len(set(teams.values()))
    output_dict = {}
    output_dict["n_clusters"] = n_clusters
    output_dict["values"] = values
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
